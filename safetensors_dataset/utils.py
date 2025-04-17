import json
from enum import Enum
from pathlib import Path
from typing import cast, MutableMapping, Mapping, Literal, Any

import torch


def get_torch_dtype_from_str(dtype: str) -> torch.dtype:
    """
    Convert the string representation of a dtype to the corresponding torch.dtype type

    :param dtype: str
    :return: torch.dtype
    """
    dtype_data = dtype.split(".")
    dtype_name = dtype_data[-1]
    return cast(torch.dtype, getattr(torch, dtype_name))


def slice_tensor(tensor: torch.Tensor, s: slice):
    if tensor.is_nested:
        dim = tensor.size(0)
        stop = min(s.stop if s.stop is not None else dim, dim)
        step = s.step if s.step is not None else 1
        return torch.nested.nested_tensor([tensor[pos] for pos in range(s.start, stop, step)])
    return tensor[s]


class TensorLayout(Enum):  # TensorStructure ?
    STANDARD = 1
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


def _map_into_dataset(
    dataset: Mapping[str, torch.Tensor | list[torch.Tensor]],
) -> MutableMapping[str, torch.Tensor]:
    map_dataset = dict()
    for key, value in dataset.items():
        if isinstance(value, list):
            if len(value) == 0:
                continue
            if len(set(map(lambda t: t.shape, value))) == 1:
                # everything has the same length, easy!
                value = torch.stack(value, dim=0)
                map_dataset[key] = value
            elif not any(map(lambda t: t.is_sparse, value)):
                value = torch.nested.nested_tensor(value)
                map_dataset[key] = value
            else:
                if not all(map(lambda t: t.is_sparse, value)):
                    raise ValueError(f"Either every element for {key} must be sparse or none")
                dims = set(map(lambda t: t.dim(), value))
                if not len(dims) == 1:
                    raise ValueError(f"{key}: mismatching dimensions: {dims}")
                batch_sizes = set(map(lambda t: t.size(0), value))
                if batch_sizes != {1}:
                    value = [elem.unsqueeze(0) for elem in value]
                batch_size = len(value)

                indices, values = list(), list()
                a = set()
                for pos, tensor in enumerate(value):
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(f"Element {pos} in {key} is not a Tensor but a {type(tensor)}")
                    if not tensor.is_coalesced():
                        tensor = tensor.coalesce()
                    tensor_indices = tensor.indices()
                    tensor_indices[0] += pos
                    if tensor_indices.numel() > 0:
                        a.add(tensor_indices[0].unique().item())
                    indices.append(tensor_indices)
                    values.append(tensor.values())
                indices = torch.cat(indices, dim=1)
                values = torch.cat(values, dim=0)

                if indices.numel() == 0:
                    max_sizes = tuple()
                    for dim in range(value[0].dim()):
                        max_sizes = max_sizes + (max(elem.size(dim) for elem in value),)
                    max_sizes = (len(value),) + max_sizes[1:]
                    value = torch.sparse_coo_tensor(indices, values, size=max_sizes)
                    map_dataset[key] = value
                    continue

                new_value = torch.sparse_coo_tensor(
                    indices,
                    values,
                    (batch_size,) + indices.amax(dim=1).add(1).unbind(dim=0)[1:],
                    is_coalesced=True
                )

                if new_value.size(0) != len(value):
                    raise ValueError(key, new_value.shape, len(value))
                map_dataset[key] = new_value
        else:
            map_dataset[key] = value
    return map_dataset


def _map_batch_into_dataset(
    dataset: MutableMapping[str, torch.Tensor],
    result: Mapping[str, torch.Tensor],
    info: Mapping[str, TensorLayout],
    batched: bool
) -> Mapping[str, TensorLayout]:
    known_layouts = dict(info)
    for key, value in result.items():
        tensor_layout = known_layouts.get(key, TensorLayout.STANDARD)
        dataset_value = dataset.get(key, None)
        if dataset_value is None:
            if tensor_layout in {TensorLayout.STANDARD, TensorLayout.VARYING_DIM_SIZE}:
                if not batched:
                    value = value.unsqueeze(0)
                if value.is_sparse and not value.is_coalesced():
                    value = value.coalesce()
                dataset[key] = value
            else:
                raise NotImplementedError(tensor_layout)  # not needed!
        else:
            if tensor_layout == TensorLayout.STANDARD:
                value = _match_dims(key, value, dataset_value, batched)

                dataset[key] = torch.cat((dataset_value, value))
            elif tensor_layout == TensorLayout.VARYING_DIM_SIZE:
                value = _match_dims(key, value, dataset_value, batched)
                if value.is_sparse:
                    if not dataset_value.is_sparse:
                        raise ValueError(f"{key}: cannot mix sparse and non sparse values")
                    value = value.coalesce()
                    indices, values = value.indices(), value.values()
                    ds_indices, ds_values = dataset_value.indices(), dataset_value.values()
                    indices[0] += ds_indices[0].max().add(1)
                    indices = torch.cat((ds_indices, indices), dim=1)
                    values = torch.cat((ds_values, values), dim=0)

                    value = torch.sparse_coo_tensor(indices, values, indices.amax(dim=1).add(1).unbind(dim=0), is_coalesced=True)
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
