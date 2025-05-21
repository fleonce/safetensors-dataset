import gc
import json
import warnings
from collections import OrderedDict, defaultdict
from functools import partial
from pathlib import Path
from typing import (
    Any, Callable, Mapping, Optional, MutableMapping, Union, Sequence, Iterable, Generic, TypeVar,
    Protocol,
)

import more_itertools
import safetensors.torch
from more_itertools import first
from tqdm import trange, tqdm
import torch
import torch.utils.data
from typing_extensions import Self

from safetensors_dataset.version import __version__
from safetensors_dataset.utils import (
    get_torch_dtype_from_str,
    TensorLayout,
    _map_batch_into_dataset, _map_into_dataset, slice_tensor, _load_safetensors_metadata, _apply_function_to_iterable,
    _maybe_wrap_index, _CHECK_INVARIANTS,
)

pack_tensor_t = dict[str, torch.Tensor]
pack_metadata_t = dict[str, Any] | None
pack_return_t = tuple[pack_tensor_t, pack_metadata_t]
can_pack_nested_tensors_fast = hasattr(torch, "_nested_view_from_buffer")


config = {
    # control whether a sharded dataset should be
    # split into multiple files, possibly improving
    # performance as file handles are not shared and
    # thus are smaller
    "shards_into_separate_files": False
}


def _check_input_dict(m: dict):
    none_keys = set()
    empty_keys = set()
    for k, v in m.items():
        if v is None:
            none_keys.add(k)
        elif isinstance(v, list):
            if len(v) == 0:
                empty_keys.add(k)

    assert not none_keys, f"Found {len(none_keys)} keys with 'None' values: {', '.join(none_keys)}"
    assert not empty_keys, f"Found {len(empty_keys)} keys with empty lists as values: {', '.join(empty_keys)}"

def _check_is_tensor(key: str, value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    raise ValueError(f"Key {key} must be a tensor, but is a {type(value)}")

def _get_items_from_tensor(key: str, tensor: torch.Tensor, indices: list[int]):
    if isinstance(tensor, Sequence) or tensor.is_nested or tensor.is_sparse:
        return [_check_is_tensor(key, tensor[i]) for i in indices]
    return tensor[indices]

def _get_len_of_item(i):
    if isinstance(i, torch.Tensor):
        return i.size(0)
    elif isinstance(i, Sequence):
        return len(i)
    raise ValueError(f"{type(i)} is unknown")


class SupportsGetItem(Protocol):
    def __getitem__(self, item: int): ...


class DatasetIterator:
    def __init__(self, dataset: SupportsGetItem):
        self.size = None
        self.dataset = dataset
        self.pos = 0

    def _size(self):
        if self.size is None:
            self.size = len(self.dataset)
        return self.size

    def __len__(self):
        return self._size()

    def __next__(self):
        if self.pos < self._size():
            try:
                return self.dataset[self.pos]
            finally:
                self.pos += 1
        raise StopIteration


class SafetensorsDataset(torch.utils.data.Dataset):
    dataset: MutableMapping[str, list[Any] | torch.Tensor]
    layout: dict[str, bool]

    def __init__(self, dataset=None, preprocess=False):
        self.dataset = _map_into_dataset(dataset or {}) if preprocess else dataset

    def __contains__(self, item: str):
        return item in self.dataset

    def shard(
        self,
        chunk_size: int = 5000,
        preprocess_if_unprocessed: bool = True,
    ) -> "ShardedSafetensorsDataset | SafetensorsDataset":
        if len(self) <= chunk_size:
            raise ValueError(f"Dataset size is smaller than chunk size ({len(self)} < {chunk_size})")

        length = len(self)
        num_chunks, remainder = divmod(length, chunk_size)
        num_chunks = num_chunks + (remainder != 0)

        chunk_datasets: tuple[dict[str, Union[torch.Tensor, list[Any]]], ...] = tuple(
            dict()
            for _ in range(num_chunks)
        )

        keys = set(self.dataset.keys())
        for key in keys:
            value = self.dataset.pop(key)
            if isinstance(value, Iterable) and not isinstance(value, torch.Tensor):
                chunks_of_lists = more_itertools.batched(value, n=chunk_size, strict=False)
                for pos, chunk in enumerate(chunks_of_lists):
                    chunk_datasets[pos][key] = chunk
            elif isinstance(tensor := value, torch.Tensor):
                is_nested = tensor.is_nested
                is_sparse = tensor.is_sparse

                if not is_nested and not is_sparse:
                    # medium easy path, just slice/chunk the tensor
                    chunks = torch.split(tensor, chunk_size, dim=0)
                    # clone here, so that we can delete the original tensor
                    chunks = tuple(chunk.clone() for chunk in chunks)
                    del tensor  # release memory
                    gc.collect()
                elif is_nested and not is_sparse:
                    if type(tensor) is torch.Tensor:
                        sizes = tensor._nested_tensor_size()
                        storage_offsets = tensor._nested_tensor_storage_offsets()
                        strides = tensor._nested_tensor_strides()

                        size_chunks = torch.split(sizes, chunk_size, dim=0)
                        stride_chunks = torch.split(strides, chunk_size, dim=0)
                        storage_offset_chunks = torch.split(storage_offsets, chunk_size, dim=0)
                        values = tensor.values()
                        chunk_value_views = tuple(
                            values[storage_offset_chunks[pos].view(-1)[0]:(storage_offset_chunks[pos + 1].view(-1)[0] if pos + 1 != num_chunks else None)].clone()
                            for pos in range(num_chunks)
                        )
                        del values
                        gc.collect()

                        # size_chunks =
                        chunks = tuple(
                            torch._nested_view_from_buffer(
                                chunk_value_views[pos],
                                size_chunk.clone(),
                                stride_chunk.clone(),
                                storage_offset_chunk.clone() - storage_offset_chunk.view(-1)[0]
                            )
                            for pos, (size_chunk, stride_chunk, storage_offset_chunk)
                            in enumerate(zip(size_chunks, stride_chunks, storage_offset_chunks))
                        )
                    else:
                        raise NotImplementedError("nested", type(tensor))
                elif is_sparse and not is_nested:
                    indices = tensor._indices()
                    values = tensor._values()
                    if indices.numel() == 0:
                        if tensor.layout == torch.sparse_coo:
                            chunks = tuple(
                                torch.sparse_coo_tensor(
                                    indices,
                                    values,
                                    size=(chunk_size if pos + 1 != num_chunks else remainder,) + tensor.shape[1:],
                                    check_invariants=_CHECK_INVARIANTS,
                                    is_coalesced=tensor.is_coalesced()
                                )
                                for pos in range(num_chunks)
                            )
                        else:
                            raise NotImplementedError(tensor.layout)
                    elif tensor.layout == torch.sparse_coo:
                        chunks = tuple()
                        indices_dim0 = indices[0]
                        for pos in range(num_chunks):
                            chunk_start = pos * chunk_size
                            chunk_end = (pos + 1) * chunk_size
                            chunk_mask = (indices_dim0 >= chunk_start) & (indices_dim0 < chunk_end)
                            chunk_indices_dim0 = indices_dim0[chunk_mask]
                            chunk_values = values[chunk_mask]
                            assert chunk_values.numel() == chunk_indices_dim0.numel()
                            if chunk_indices_dim0.numel() == 0:
                                raise NotImplementedError

                            chunk_indices = indices[:, chunk_mask]
                            chunk_indices[0] -= chunk_start
                            chunks = chunks + (torch.sparse_coo_tensor(
                                chunk_indices.clone(),
                                chunk_values.clone(),
                                (chunk_size if pos + 1 != num_chunks else remainder,) + tensor.shape[1:],
                                check_invariants=_CHECK_INVARIANTS,
                                is_coalesced=tensor.is_coalesced()
                            ),)
                        del tensor
                        del indices
                        del values
                        gc.collect()
                    else:
                        raise NotImplementedError(tensor.layout)
                else:
                    raise ValueError(f"Tensor cannot be nested and sparse")

                for pos in range(num_chunks):
                    chunk_datasets[pos][key] = chunks[pos]
            else:
                raise ValueError(f"value in dataset must be an Iterable or torch.Tensor, but got {type(value)}")
        return ShardedSafetensorsDataset(tuple(
            SafetensorsDataset(chunk, preprocess=preprocess_if_unprocessed)
            for chunk in chunk_datasets
        ))

    def filter(
        self,
        filter_fn: Callable[[dict[str, torch.Tensor]], bool],
        use_tqdm: bool = True
    ):
        filtered_dataset = dict({k: list() for k in self.dataset.keys()})
        from_to = range if not use_tqdm else partial(trange, leave=False)
        for i in from_to(len(self)):
            elem = self.__getitem__unsafe(i)
            if filter_fn(elem):
                for k in filtered_dataset.keys():
                    filtered_dataset[k].append(elem[k])
        return SafetensorsDataset(filtered_dataset)

    def pack(self) -> Self:
        for key in self.keys():
            if isinstance(self[key], list):
                if any(elem.is_sparse for elem in self[key]):
                    self.dataset[key] = torch.stack(self.dataset[key], dim=0).coalesce()
                    continue
                self.dataset[key] = torch.nested.nested_tensor(self.dataset[key])
        return self

    def keys(self) -> set[str]:
        return set(self.dataset.keys())

    def __getitem__unsafe(self, item: int | str):
        if isinstance(item, str):
            return self.dataset[item]
        return {k: v[item] for k, v in self.dataset.items()}

    def __getitem__(self, item: int | str) -> dict[str, torch.Tensor] | torch.Tensor:
        if isinstance(item, str):
            return self.dataset[item]
        return {k: _check_is_tensor(k, v[item]) for k, v in self.dataset.items()}

    def __getitems__(self, indices: list[int]):
        elements_per_key = {k: _get_items_from_tensor(k, v, indices) for k, v in self.dataset.items()}
        return [{k: elements_per_key[k][i] for k in elements_per_key.keys()} for i in range(len(indices))]

    def __len__(self):
        return next((_get_len_of_item(v) for v in self.dataset.values()), 0)

    def __iter__(self):
        return DatasetIterator(self)

    @property
    def device(self) -> torch.device:
        if len(self) == 0:
            raise ValueError("Cannot determine device in empty dataset")
        return next(iter(self.dataset.values())).device

    def to(self, device: torch.device | str | int) -> "SafetensorsDataset":
        self_device = self.device
        if self_device == device:
            return self

        device_dataset = {key: value.to(device) for key, value in self.dataset.items()}
        return self.__class__(device_dataset)

    def map(
        self,
        func: Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
        info: Optional[Mapping[str, TensorLayout]] = None,
        strict: bool = False,
        use_tqdm: bool = True,
        batched: bool = False,
        batch_size: int = 1,
    ) -> "SafetensorsDataset":
        items = self._transpose(batched, batch_size)
        dataset = _apply_function_to_iterable(
            func,
            items,
            len(self),
            batched=batched,
            batch_size=batch_size,
            disable_tqdm=not use_tqdm,
        )

        return self.__class__(dataset, preprocess=False)

    def select(self, indices: list[int], use_tqdm=False) -> "SafetensorsDataset":
        select_dataset: MutableMapping[str, torch.Tensor] = {}
        info = self.info()
        for pos in tqdm(indices, disable=not use_tqdm, total=len(indices)):
            entry = {k: v.clone() for k, v in self[pos].items()}

            _map_batch_into_dataset(select_dataset, entry, info, False)
        return self.__class__(select_dataset)

    def info(self) -> Mapping[str, TensorLayout]:
        def tensor_layout_for_key(k: str):
            if isinstance(self.dataset[k], list):
                if not isinstance(first(self.dataset[k]), torch.Tensor):
                    return TensorLayout.NO_TENSOR
                shapes = set(map(lambda x: x.shape, self.dataset[k]))
                if len(shapes) > 1:
                    return TensorLayout.VARYING_DIM_SIZE
                return TensorLayout.STANDARD
            elif not isinstance(self.dataset[k], torch.Tensor):
                return TensorLayout.NO_TENSOR
            elif self.dataset[k].is_nested or self.dataset[k].is_sparse:
                return TensorLayout.VARYING_DIM_SIZE
            return TensorLayout.STANDARD
        return {
            key: tensor_layout_for_key(key)
            for key in self.keys()
        }

    def rename(self, key: str, new_key: str):
        self.dataset[new_key] = self.dataset[key]

    def __add__(self, other: "SafetensorsDataset") -> "SafetensorsDataset":
        elements = dict(self.dataset)
        for key in other.keys():
            if key in elements:
                raise ValueError(f"Duplicate key {key}")
            elements[key] = other[key]
        return SafetensorsDataset(elements)

    def __iadd__(self, other: "SafetensorsDataset"):
        for key in other.keys():
            if key in self:
                raise ValueError(f"Duplicate key {key}")
            self.dataset[key] = other.dataset[key]

    def _transpose(self, batched=False, batch_size=0):
        keys = self.keys()
        if batched:
            pos = 0
            for i in range(0, len(self), batch_size):
                pos += batch_size
                yield {
                    key: slice_tensor(self.dataset[key], slice(i, i+batch_size, None))
                    for key in keys
                }
            if pos < len(self):
                yield {
                    key: slice_tensor(self.dataset[key], slice(pos, None, None))
                    for key in keys
                }
        else:
            for i in range(len(self)):
                yield {
                    key: self.dataset[key][i] for key in keys
                }

    def __repr__(self):
        def nice_shape(shape):
            return "[" + " x ".join(map(str, shape)) + "]"

        def nested_shape(elem):
            elem_type = type(elem).__name__
            if elem is None:
                return tuple()
            elif isinstance(elem, str):
                return (elem_type,)
            elif isinstance(elem, torch.Tensor):
                return tuple(elem.shape)
            elif isinstance(elem, Iterable):
                return (elem_type,) + nested_shape(first(elem, None))
            return (elem_type,)

        def shape_for_elem(elem):
            is_tensor = isinstance(elem, torch.Tensor)
            if is_tensor and not elem.is_nested:
                return nice_shape(elem.shape)
            elif isinstance(elem, Sequence) or (is_tensor and elem.is_nested):
                shape = (len(elem) if not is_tensor else elem.size(0), )
                inner_shape = None
                for list_elem in elem:
                    if not isinstance(list_elem, torch.Tensor):
                        inner_shape = nested_shape(list_elem)
                        break
                    inner_shape = inner_shape or list_elem.shape
                    inner_shape = tuple(map(max, inner_shape, list_elem.shape))
                shape = shape + inner_shape
                return nice_shape(shape)
            else:
                raise ValueError(f"Unknown element type {type(elem)}")
        shapes = list()
        shapes_of_keys = defaultdict(list)
        for key, value in self.dataset.items():
            shapes_of_keys[shape_for_elem(value)].append(key)
        comma = ", "
        for shape, keys in shapes_of_keys.items():
            shapes.append(comma.join(keys) + "=" + shape)
        comma_lf = ",\n  "
        return f"SafetensorsDataset(\n  {comma_lf.join(shapes)}\n)"

    def __del__(self):
        del self.dataset

    @staticmethod
    def unpack_list_tensor(key: str, metadata: Mapping[str, Any], meta: Mapping[str, Any], storage: Mapping[str, torch.Tensor]):
        numel = meta.get("numel")
        tensors = list()
        for elem in range(numel):
            tensors.append(storage.get(key + str(elem)))
        return torch.nested.nested_tensor(tensors)

    @staticmethod
    def unpack_nested_tensor(key: str, metadata: Mapping[str, Any], meta: Mapping[str, Any], storage: Mapping[str, torch.Tensor]):
        buffer = storage[key + ".buffer"]
        sizes = storage[key + ".sizes"]
        if key + ".strides" in storage:
            strides = storage[key + ".strides"]
        else:
            strides = torch.ones_like(sizes)
        if key + ".storage_offsets" in storage:
            storage_offsets = storage[key + ".storage_offsets"]
        else:
            storage_offsets = sizes.cumsum(dim=0).roll(1).squeeze()
            storage_offsets[0] = 0
        tensor = torch._nested_view_from_buffer(buffer, sizes, strides, storage_offsets)
        return tensor

    @staticmethod
    def unpack_sparse_tensor(key: str, metadata: Mapping[str, Any], meta: Mapping[str, Any], storage: Mapping[str, torch.Tensor]):
        numel = meta.get("numel")
        if not numel:
            numel = metadata.get("size")
        dims = meta.get("dims")
        if numel != dims[0]:
            dims = (numel,) + tuple(dims)
        dtype = meta.get("dtype")
        dtype = get_torch_dtype_from_str(dtype)
        if dtype == torch.bool and key + ".indices" not in storage:
            tensor = storage[key]
            tensor = torch.sparse_coo_tensor(tensor, torch.ones(tensor.size(-1), dtype=dtype), size=tuple(dims), check_invariants=_CHECK_INVARIANTS)
            tensor = tensor.coalesce()
        else:
            indices = storage[key + ".indices"]
            if key + ".values" not in storage:
                raise ValueError(f"Need {key}.values to restore sparsely stored tensor")
            values = storage[key + ".values"]
            tensor = torch.sparse_coo_tensor(indices, values, size=dims, check_invariants=_CHECK_INVARIANTS)
            tensor = tensor.coalesce()
        return tensor

    @staticmethod
    def pack_single_tensor(key: str, tensor: torch.Tensor) -> pack_return_t:
        pack: pack_tensor_t
        metadata: pack_metadata_t
        if tensor.is_sparse:
            if tensor.dtype == torch.bool:
                pack = {
                    key: tensor._indices()
                }
            else:
                pack = {
                    key + ".values": tensor._values(),
                    key + ".indices": tensor._indices()
                }
            metadata = {
                "sparse": True,
                "dtype": repr(tensor.dtype),
                "dims": tensor.shape,
                "numel": tensor.size(0),
            }
            return pack, metadata
        elif tensor.is_nested:
            if not can_pack_nested_tensors_fast:
                raise ValueError(
                    "To efficiently store and load nested tensors, a recent version of pytorch >= 2.1.0 is required."
                )
            buffer = tensor.values()
            pack = {
                key + ".buffer": buffer,
                key + ".sizes": tensor._nested_tensor_size()
            }
            if tensor.dim() > 2:
                pack[key + ".strides"] = tensor._nested_tensor_strides()
                pack[key + ".storage_offsets"] = tensor._nested_tensor_storage_offsets()
            metadata = {
                "nested": True,
                "dtype": repr(tensor.dtype),
                "numel": tensor.size(0),
            }
            return pack, metadata
        return {key: tensor}, None

    def pack_tensor_list(self, key: str, tensors: Sequence[torch.Tensor]) -> pack_return_t:
        if len(tensors) == 0:
            raise ValueError(f"Cannot save an empty list of tensors for key '{key}'")
        if not isinstance(tensors[0], torch.Tensor):
            raise ValueError(f"Elements of '{key}' are no tensors ... element 0 is {type(tensors[0])}")
        if len(tensors) != len(self):
            raise ValueError(f"'{key}' should have {len(self)} elements, but has {len(tensors)}")
        if not isinstance(tensors, list):
            tensors = list(tensors)

        dims = set(map(torch.Tensor.dim, tensors))  # noqa
        if max(dims) > min(dims):
            raise ValueError(f"Elements of '{key}' are of different dimensionality")

        is_sparse = any(map(lambda x: x.is_sparse, tensors))
        is_nested = any(map(lambda x: x.is_nested, tensors))
        if is_sparse and is_nested:
            raise ValueError(f"Cannot pack a mixed list of sparse and nested tensors for key '{key}'")

        if is_sparse:
            sparse_shape = tuple(max(map(lambda x: x.size(dim), tensors)) for dim in range(min(dims)))
            same_size_tensors = list(map(lambda t: torch.sparse_coo_tensor(t._indices(), t._values(), size=sparse_shape, check_invariants=_CHECK_INVARIANTS), tensors))
            sparse_tensor = torch.stack(same_size_tensors, dim=0).coalesce()
            if sparse_tensor.dtype == torch.bool:
                pack = {key: sparse_tensor._indices()}
            else:
                pack = {key + ".indices": sparse_tensor._indices(), key + ".values": sparse_tensor._values()}
            metadata = {
                "sparse": True,
                "dims": sparse_tensor.shape,
                "dtype": repr(sparse_tensor.dtype),
                "numel": len(tensors)
            }
            return pack, metadata

        if not can_pack_nested_tensors_fast:
            warnings.warn(
                f"To efficiently store and load nested tensors, a recent version of pytorch >= 2.1.0 is required, "
                f"you are on {torch.__version__}"
            )
            pack = dict()
            for index, elem in enumerate(tensors):
                pack[key + "." + str(index)] = elem
            metadata = {"list": True, "numel": len(tensors)}
            return pack, metadata
        else:
            if is_nested:
                nested_tensor = torch.stack(tensors)
            else:
                nested_tensor = torch.nested.nested_tensor(tensors)
            pack = {
                key + ".buffer": nested_tensor.values(),
                key + ".sizes": nested_tensor._nested_tensor_size()
            }
            if nested_tensor.dim() > 2 or nested_tensor._nested_tensor_size().numel() == 0:
                pack = pack | {
                    key + ".strides": nested_tensor._nested_tensor_strides(),
                    key + ".storage_offsets": nested_tensor._nested_tensor_storage_offsets()
                }
            metadata = {"nested": True, "numel": len(tensors)}
            return pack, metadata

    def _save_to_dict(self):
        def check_key(key: str):
            if "." in key:
                raise ValueError(f". is not allowed in a safetensors dataset (used in {key})")

        metadata = {"size": len(self), "version": __version__}
        tensors = OrderedDict()
        for k, v in self.dataset.items():
            check_key(k)

            pack, pack_metadata = None, None
            if isinstance(v, torch.Tensor):
                pack, pack_metadata = self.pack_single_tensor(k, v)
            elif isinstance(v, Sequence):
                pack, pack_metadata = self.pack_tensor_list(k, v)
            else:
                raise ValueError(f"Cannot pack value type {type(v)} for key {k}")

            if pack is not None:
                tensors.update(pack)
                if pack_metadata is not None:
                    metadata[k] = pack_metadata

        metadata = {k: json.dumps(v) for k, v in metadata.items()}
        return tensors, metadata

    def save_to_file(self, path: Union[str, Path]):
        tensors, metadata = self._save_to_dict()
        safetensors.torch.save_file(tensors, path, metadata=metadata)

    @classmethod
    def _load_from_dict(cls, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]):
        dataset = {}
        keys = set()
        for k in tensors.keys():
            if "." in k:
                # a key in the dataset may be stored with multiple keys in the underlying structure
                # f. e. '.values' and '.indices'
                info = k.split(".")
                keys.add('.'.join(info[:-1]))
            else:
                keys.add(k)

        for k in keys:
            meta: Mapping[str, Any] = metadata.get(k, dict())
            if not meta:
                # load a single tensor
                tensor = tensors[k]
            elif meta.get("sparse", False) is True:
                tensor = cls.unpack_sparse_tensor(k, metadata, meta, tensors)
            elif meta.get("nested", False) is True:
                tensor = cls.unpack_nested_tensor(k, metadata, meta, tensors)
            elif meta.get("list", False) is True:
                tensor = cls.unpack_list_tensor(k, metadata, meta, tensors)
            else:
                raise ValueError(f"Cannot unpack stored tensor {k} with metadata = {meta}")
            dataset[k] = tensor
        return SafetensorsDataset(dataset)

    @classmethod
    def load_from_file(cls, path: Path):
        metadata = _load_safetensors_metadata(path)
        tensors = safetensors.torch.load_file(path, device="cpu")
        return cls._load_from_dict(tensors, metadata)

    @classmethod
    def from_dict(cls, x: dict[str, torch.Tensor | list[torch.Tensor]], preprocess: bool=False):
        _check_input_dict(x)
        return SafetensorsDataset(x, preprocess)

    @classmethod
    def from_list(cls, x: list[dict[str, torch.Tensor]], preprocess: bool=False):
        out = {}
        for i in x:
            for k, v in i.items():
                if k not in out:
                    out[k] = [v]
                else:
                    out[k].append(v)
        return cls.from_dict(out, preprocess=preprocess)

class ShardedSafetensorsDataset(torch.utils.data.Dataset):
    shards: tuple[SafetensorsDataset, ...]

    def __init__(self, shards: tuple[SafetensorsDataset, ...]):
        self.shards: tuple[SafetensorsDataset, ...] = shards
        self.shard_size = len(shards[0])

    def __contains__(self, item):
        return item in self.shards[0]

    def __len__(self):
        return (len(self.shards) - 1) * self.shard_size + (len(self.shards[-1]) if len(self.shards) > 1 else 0)

    def __getitem__(self, item: int | str) -> dict[str, torch.Tensor] | torch.Tensor:
        if isinstance(item, str):
            raise NotImplementedError(f"Cannot access keys for sharded datasets")
        item = _maybe_wrap_index(item, len(self))
        shard = item // self.shard_size
        if shard >= len(self.shards) or item >= len(self):
            raise IndexError(item)
        dataset_shard = self.shards[shard].dataset
        item %= self.shard_size
        return {k: v[item] for k, v in dataset_shard.items()}

    def __repr__(self):
        lines = [f"ShardedSafetensorsDataset(size={len(self)}, shard_size={self.shard_size}, num_shards={len(self.shards)},\n"]
        lf = "\n"
        lfspace = "\n  "
        for shard in self.shards:

            lines.append(f"  {repr(shard).replace(lf, lfspace)}\n")
        lines.append(")")
        return "".join(lines)

    def get_shard(self, pos: Optional[int] = None) -> SafetensorsDataset:
        if pos is None:
            raise NotImplementedError("Cannot shard() a sharded dataset")
        return self.shards[pos]

    def __getitems__(self, indices: list[int]):
        buckets: MutableMapping[int, list[int]] = dict()
        size = len(self)
        wrapped_indices = list()
        for index in indices:
            wrapped_index = _maybe_wrap_index(index, size)
            if wrapped_index >= size:
                raise IndexError(f"Index {index} ({wrapped_index}) is out of range for {self}")
            wrapped_indices.append(wrapped_index)
            bucket = wrapped_index // self.shard_size
            if bucket not in buckets:
                buckets[bucket] = list()
            buckets[bucket].append(wrapped_index % self.shard_size)

        bucket_items: MutableMapping[int, list[dict[str, torch.Tensor]]] = dict()
        for bucket, bucket_indices in buckets.items():
            bucket_items[bucket] = self.get_shard(bucket).__getitems__(bucket_indices)

        return [
            bucket_items[index // self.shard_size][buckets[index // self.shard_size].index(index % self.shard_size)]
            for index in wrapped_indices
        ]

    def save_to_file(self, path: Union[str, Path]):
        tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        metadata: dict[str, Any] = {"num_shards": str(len(self.shards))}
        for pos, shard in enumerate(self.shards):
            shard_tensors, shard_metadata = shard._save_to_dict()

            for key, tensor in shard_tensors.items():
                tensors[f"shards.{pos}.{key}"] = tensor

            for key, value in shard_metadata.items():
                metadata[f"shards.{pos}.{key}"] = value
        safetensors.torch.save_file(tensors, path, metadata=metadata)

    @classmethod
    def _load_from_dict(cls, tensors, metadata):
        if "num_shards" not in metadata:
            raise ValueError("num_shards")
        num_shards = int(metadata["num_shards"])

        shard_datasets = tuple()
        for pos in range(num_shards):
            shard_tensors = {
                key[len(shard_prefix):]: value
                for key, value in tensors.items()
                if key.startswith(shard_prefix := f"shards.{pos}.")
            }

            shard_metadata = {
                key[len(shard_prefix):]: value
                for key, value in metadata.items()
                if key.startswith(shard_prefix := f"shards.{pos}.")
            }

            shard_dataset = SafetensorsDataset._load_from_dict(shard_tensors, shard_metadata)
            shard_datasets = shard_datasets + (shard_dataset,)
        return ShardedSafetensorsDataset(shard_datasets)

    @classmethod
    def load_from_file(cls, path: Union[str, Path]):
        metadata = _load_safetensors_metadata(path)
        tensors = safetensors.torch.load_file(path, device="cpu")
        return cls._load_from_dict(tensors, metadata)
