import inspect
import json
from enum import Enum
from pathlib import Path
from typing import cast, MutableMapping, Mapping, Any, Sequence, Union, Generator

import torch
from more_itertools import first
from more_itertools.recipes import flatten
from tqdm import tqdm


def get_torch_dtype_from_str(dtype: str) -> torch.dtype:
    """
    Convert the string representation of a dtype to the corresponding torch.dtype type

    :param dtype: str
    :return: torch.dtype
    """
    dtype_data = dtype.split(".")
    dtype_name = dtype_data[-1]
    return cast(torch.dtype, getattr(torch, dtype_name))


def slice_tensor(tensor: Any, s: slice):
    if not isinstance(tensor, torch.Tensor):
        return tensor[s]

    if tensor.is_nested:
        dim = tensor.size(0)
        stop = min(s.stop if s.stop is not None else dim, dim)
        step = s.step if s.step is not None else 1
        return torch.nested.nested_tensor([tensor[pos] for pos in range(s.start, stop, step)])
    return tensor[s]


class TensorLayout(Enum):  # TensorStructure ?
    STANDARD = 1
    NO_TENSOR = 2
    VARYING_DIM_SIZE = 3


def try_size(t: torch.Tensor, dim: int):
    try:
        return t.size(dim)
    except RuntimeError:
        pass
    return f"s{dim}"


def nt_size(t: torch.Tensor):
    return tuple(try_size(t, i) for i in range(t.dim()))


def _load_safetensors_metadata(fp: str | Path) -> dict[str, Any]:
    with open(fp, 'rb') as f:
        n_bytes = f.read(8)
        n_bytes = int.from_bytes(n_bytes, byteorder='little', signed=False)
        content = f.read(n_bytes)
        content = content.decode("utf-8")
        metadata = json.loads(content)['__metadata__']
        metadata = {k: json.loads(v) for k, v in metadata.items()}
        return metadata

_CHECK_INVARIANTS = False

def _concat_sparse_tensors_of_different_shapes(tensors: Sequence[torch.Tensor], batched: bool):
    if not batched:
        tensors = [tensor.unsqueeze(0) for tensor in tensors]

    pos, numel = 0, 0
    indices, values = list(), list()
    max_sizes = (len(tensors),) + (0,) * (tensors[0].dim() - 1)
    is_coalesced = True
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Element {pos} is not a Tensor but a {type(tensor)}")

        tensor_indices = tensor._indices()  # avoid coalesc'ing check
        tensor_values = tensor._values()
        is_coalesced = is_coalesced and tensor.is_coalesced()

        tensor_indices[0] += pos
        pos += tensor.size(0)
        numel += tensor_values.numel()
        indices.append(tensor_indices)
        values.append(tensor_values)

        max_sizes = tuple(max(tensor_size, max_size) for tensor_size, max_size in zip(tensor.shape, max_sizes))

    if numel > 0:
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values, dim=0)
    else:
        indices = tensors[0].new_empty((len(max_sizes), 0), dtype=torch.long, layout=torch.strided)
        values = tensors[0].new_empty((0,), layout=torch.strided)

    return torch.sparse_coo_tensor(
        indices,
        values,
        size=max_sizes,
        is_coalesced=is_coalesced or numel == 0,
        check_invariants=_CHECK_INVARIANTS
    )


def _maybe_wrap_index(pos: int, size: int) -> int:
    if pos < 0:
        return size + pos
    return pos


def _apply_function_to_iterable(
    func,
    iterable,
    numel: int,
    batched: bool,
    batch_size: int,
    disable_tqdm: bool = False,
):
    def _collect_output(
        output: Union[Sequence[Mapping[str, Any]], Generator[Mapping[str, Any], None, None], Mapping[str, Any]],
        target: MutableMapping[str, list[Any]],
    ):
        if isinstance(output, bool) and output is False:
            return
        elif (
            isinstance(output, Sequence)
            or inspect.isgenerator(output)
        ):
            for element in output:
                _collect_output(element, target)
        else:
            for key, value in output.items():
                if key not in target:
                    target[key] = list()
                target[key].append(value)

    out = {}
    done = 0
    desc = getattr(func, "__name__", None)
    with tqdm(desc=desc, disable=disable_tqdm, total=numel) as progress_bar:
        for pos, item in enumerate(iterable):
            _collect_output(func(item), out)

            new_done = done + (not batched or batch_size)
            progress = new_done - done - (new_done % numel if new_done > numel else 0)
            progress_bar.update(progress)
            done += progress

    return _map_into_dataset(out, batched=batched)


def _batched_map_into_dataset(
    dataset: Mapping[str, list[torch.Tensor] | tuple[torch.Tensor, ...]]
) -> MutableMapping[str, torch.Tensor]:
    map_dataset = dict()
    for key, value in dataset.items():
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue

            first_value = first(value)
            if not isinstance(first_value, torch.Tensor):
                map_dataset[key] = list(flatten(value))
                continue
            is_nested = (
                first_value.is_nested
            )
            if not is_nested:
                if len(set(map(lambda t: t.shape[1:], value))) == 1:
                    tensor = torch.cat(value, dim=0)
                elif not any(map(lambda t: t.is_sparse, value)):
                    if not isinstance(value, list):
                        value = list(value)
                    tensor = torch.nested.nested_tensor(value)
                else:
                    tensor = _concat_sparse_tensors_of_different_shapes(value, batched=True)
            else:
                tensor = torch.cat(value, dim=0)
            map_dataset[key] = tensor
        elif isinstance(value, torch.Tensor):
            map_dataset[key] = value
        else:
            raise ValueError(f"Keys must be tuple, list or torch.Tensor, got {type(value)} for {key}")
    return map_dataset

def _map_into_dataset(
    dataset: Mapping[str, torch.Tensor | list[torch.Tensor]],
    batched: bool = False,
) -> MutableMapping[str, torch.Tensor]:
    if batched:
        return _batched_map_into_dataset(dataset)
    map_dataset = dict()
    for key, value in dataset.items():
        if isinstance(value, list):
            if len(value) == 0:
                continue
            if not isinstance(first(value), torch.Tensor):
                map_dataset[key] = value
                continue
            if len(set(map(lambda t: t.shape, value))) == 1:
                # everything has the same length, easy!
                value = torch.stack(value, dim=0)
                map_dataset[key] = value
            elif not any(map(lambda t: t.is_sparse, value)):
                value = torch.nested.nested_tensor(value)
                map_dataset[key] = value
            else:
                new_value = _concat_sparse_tensors_of_different_shapes(value, batched=False)
                map_dataset[key] = new_value
        else:
            map_dataset[key] = value
    return map_dataset


def _map_batch_into_dataset(
    dataset: MutableMapping[str, torch.Tensor | list[Any]],
    result: Mapping[str, torch.Tensor],
    info: Mapping[str, TensorLayout],
    batched: bool,
    strict: bool = False,
) -> Mapping[str, TensorLayout]:
    known_layouts = dict(info)
    for key, value in result.items():
        tensor_layout = known_layouts.get(key, TensorLayout.STANDARD)
        dataset_value = dataset.get(key, None)
        if not isinstance(value, torch.Tensor):
            if (
                isinstance(value, (list, tuple))
                and len(value) > 0
                and isinstance(first(value), torch.Tensor)
            ):
                if dataset_value is None:
                    if len(set(map(lambda t: t.shape, value))) == 1:
                        dataset[key] = torch.stack(value, dim=0)
                    elif first(value).is_sparse:
                        raise NotImplementedError("sparse lists")
                    else:
                        dataset[key] = torch.nested.nested_tensor(value)
                elif dataset_value.is_sparse:
                    raise NotImplementedError()
                elif dataset_value.is_nested:
                    value = torch.nested.nested_tensor(value)

                    dataset[key] = torch.cat((dataset_value, value), dim=0)
                else:
                    if len(set(map(lambda t: t.shape, value))) == 1:
                        value = torch.stack(value, dim=0)
                    elif first(value).is_sparse:
                        raise NotImplementedError("sparse lists")
                    else:
                        dataset_value = torch.nested.as_nested_tensor(dataset_value)
                        value = torch.nested.nested_tensor(value)
                    dataset[key] = torch.cat((dataset_value, value), dim=0)
            else:
                if strict:
                    raise ValueError(f"{key} must be a torch.Tensor, got a {type(value)}")
                else:
                    if dataset_value is not None and not isinstance(dataset_value, list):
                        raise ValueError(f"{key} must be a {type(value)}, previously got a torch.Tensor")
                    elif dataset_value is None:
                        dataset[key] = [value]
                    else:
                        dataset_value.append(value)
            continue
        if dataset_value is None:
            if tensor_layout in {TensorLayout.STANDARD, TensorLayout.VARYING_DIM_SIZE}:
                if not batched:
                    value = value.unsqueeze(0)
                if value.is_sparse and not value.is_coalesced():
                    value = value.coalesce()
                dataset[key] = value
            elif tensor_layout == TensorLayout.NO_TENSOR:
                dataset[key] = [value]
            else:
                raise NotImplementedError(tensor_layout)  # not needed!
        else:
            if not isinstance(dataset_value, torch.Tensor):
                raise ValueError(f"dataset_value must be a torch.Tensor, got {type(dataset_value)}")

            if tensor_layout in {TensorLayout.STANDARD, TensorLayout.VARYING_DIM_SIZE}:
                value = _match_dims(key, value, dataset_value, batched)
                if dataset_value.is_nested or (value.dim() > 1 and value.size(1) != (
                    dataset_value.size(1) if not dataset_value.is_nested
                    else max(map(lambda t: t.size(0), dataset_value.unbind(0)))
                )):
                    tensor_layout = TensorLayout.VARYING_DIM_SIZE
            if tensor_layout == TensorLayout.STANDARD:
                value = torch.cat((dataset_value, value))
                if dataset_value.is_sparse:
                    value = value.coalesce()
                dataset[key] = value
            elif tensor_layout == TensorLayout.VARYING_DIM_SIZE:
                if value.is_sparse:
                    sizes = (dataset_value.size(0) + value.size(0),)
                    for dim in range(1, value.dim()):
                        sizes = sizes + (max(value.size(dim), dataset_value.size(dim)),)
                    value = value.coalesce()
                    indices, values = value._indices(), value._values()
                    ds_indices, ds_values = dataset_value._indices(), dataset_value._values()
                    if ds_indices.numel() == indices.numel() == 0:
                        value = torch.sparse_coo_tensor(indices, values, size=sizes, check_invariants=_CHECK_INVARIANTS)
                    else:
                        indices[0] += dataset_value.size(0)
                        indices = torch.cat((ds_indices, indices), dim=1)
                        values = torch.cat((ds_values, values), dim=0)
                        value = torch.sparse_coo_tensor(indices, values, size=sizes, is_coalesced=True, check_invariants=_CHECK_INVARIANTS)
                    dataset[key] = value
                else:
                    if not dataset_value.is_nested:
                        if not batched:
                            if (
                                not value.is_nested
                                and dataset_value.shape[1:] == value.shape[1:]
                            ):
                                # fast way, we can just cat
                                value = torch.cat((dataset_value, value), dim=0)
                                dataset[key] = value
                            elif not value.is_nested:
                                # => ds.shape[1:] != value.shape[1:]
                                dataset_value = torch.nested.as_nested_tensor(dataset_value)
                                value = torch.nested.as_nested_tensor(value)
                                value = torch.cat((dataset_value, value), dim=0)
                                dataset[key] = value
                            else:
                                raise ValueError(f"{key} cannot be nested in {batched=}")
                        elif batched:
                            if (
                                not value.is_nested
                                and dataset_value.shape[1:] == value.shape[1:]
                            ):
                                value = torch.cat((dataset_value, value), dim=0)
                                dataset[key] = value
                            else:
                                if not value.is_nested:
                                    value = torch.nested.as_nested_tensor(value)
                                dataset_value = torch.nested.as_nested_tensor(dataset_value)
                                value = torch.cat((dataset_value, value), dim=0)
                                dataset[key] = value
                        else:
                            raise NotImplementedError(batched)
                    else:
                        if not batched:
                            if value.is_nested:
                                raise ValueError(f"{key} cannot be nested in {batched=}")
                            else:
                                value = torch.nested.as_nested_tensor(value)
                                value = torch.cat((dataset_value, value), dim=0)
                                dataset[key] = value
                        elif batched:
                            if not value.is_nested:
                                value = torch.nested.as_nested_tensor(value)
                            value = torch.cat((dataset_value, value), dim=0)
                            dataset[key] = value
                        else:
                            raise NotImplementedError(batched)
            elif tensor_layout == TensorLayout.NO_TENSOR:
                dataset_value.append(value)
            else:
                raise NotImplementedError(tensor_layout)


def _match_dims(key: str, tensor: torch.Tensor, match: torch.Tensor, batched: bool) -> torch.Tensor:
    if not batched:
        if match.dim() == tensor.dim():
            if tensor.size(0) != 1:
                raise ValueError(f"{key} was returned with shape {tensor.shape} but dataset has {match.shape}")
        elif match.dim() < tensor.dim():
            raise ValueError(f"{key} is of larger dim than dataset: {tensor.shape} vs {match.shape}")
        else:
            if match.dim() != tensor.dim() + 1:
                raise ValueError(
                    f"Cannot unsqueeze(dim=0) on returned value for {key} to match dataset: "
                    f"{tensor.shape} vs {match.shape}"
                )
            return tensor.unsqueeze(0)
        return tensor
    elif batched:
        if match.dim() != tensor.dim():
            raise ValueError(f"Got different dimensions for {key} between batch ({tensor.shape}) and dataset ({match.shape})")
        return tensor
    else:
        raise ValueError(batched)
