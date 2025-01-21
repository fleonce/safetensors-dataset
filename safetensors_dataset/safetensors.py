import json
import warnings
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, MutableMapping

import safetensors.torch
from tqdm import trange, tqdm
import torch
import torch.utils.data
from typing_extensions import Self

from safetensors_dataset.version import __version__
from safetensors_dataset.utils import (
    get_torch_dtype_from_str,
    TensorLayout,
    _map_batch_into_dataset, _map_into_dataset,
)

pack_tensor_t = dict[str, torch.Tensor]
pack_metadata_t = dict[str, Any] | None
pack_return_t = tuple[pack_tensor_t, pack_metadata_t]
can_pack_nested_tensors_fast = hasattr(torch, "_nested_view_from_buffer")


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

def _get_items_from_tensor(t: torch.Tensor, indices: list[int, ...]):
    if isinstance(t, list) or t.is_nested or t.is_sparse:
        return [t[i] for i in indices]
    return t[indices]


class SafetensorsDataset(torch.utils.data.Dataset):
    dataset: MutableMapping[str, list[torch.Tensor] | torch.Tensor]
    layout: dict[str, bool]

    def __init__(self, dataset=None, preprocess=False):
        self.dataset = _map_into_dataset(dataset or {}) if preprocess else dataset

    def filter(
        self,
        filter_fn: Callable[[dict[str, torch.Tensor]], bool],
        use_tqdm: bool = True
    ):
        filtered_dataset = dict({k: list() for k in self.dataset.keys()})
        from_to = range if not use_tqdm else partial(trange, leave=False)
        for i in from_to(len(self)):
            elem = self[i]
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

    def __getitem__(self, item: int | str) -> dict[str, torch.Tensor] | torch.Tensor:
        if isinstance(item, str):
            return self.dataset[item]
        return {k: v[item] for k, v in self.dataset.items()}

    def __getitems__(self, indices: list[int, ...]):
        elements_per_key = {k: _get_items_from_tensor(v, indices) for k, v in self.dataset.items()}
        return [{k: elements_per_key[k][i] for k in elements_per_key.keys()} for i in range(len(indices))]

    def __len__(self):
        return next((self._get_len_of_item(v) for v in self.dataset.values()), 0)

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
        use_tqdm: bool = True,
        batched: bool = False,
        batch_size: int = 1,
    ) -> "SafetensorsDataset":
        info = info or dict()
        map_dataset: MutableMapping[str, torch.Tensor] = {}
        items = self._transpose(batched, batch_size)
        done = 0
        n_elem = len(self)
        with tqdm(disable=not use_tqdm, total=len(self)) as progress_bar:
            for pos, item in enumerate(items):
                transformed_item = func(item)
                for key in transformed_item.keys():
                    if pos > 0 and key not in map_dataset:
                        raise ValueError(key + " was added after the first item")

                _map_batch_into_dataset(
                    map_dataset,
                    transformed_item,
                    info,
                    batched
                )
                new_done = done + (not batched or batch_size)
                progress = new_done - done - (new_done % n_elem if new_done > n_elem else 0)
                progress_bar.update(progress)
                done += progress

        for key in self.keys():
            tensor_layout = info.get(key, TensorLayout.STANDARD)
            if tensor_layout == TensorLayout.VARYING_DIM_SIZE:
                if not map_dataset[key].is_nested and not map_dataset[key].is_sparse:
                    warnings.warn(f"{tensor_layout} was specified for {key} but shape is consistent across all elements")

        return self.__class__(map_dataset)

    def _transpose(self, batched=False, batch_size=0):
        keys = self.keys()
        if batched:
            pos = 0
            for i in range(0, len(self), batch_size):
                pos += batch_size
                yield {
                    key: self.dataset[key][i:i+batch_size]
                    for key in keys
                }
            if pos < len(self):
                yield {
                    key: self.dataset[key][pos:]
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

        def shape_for_elem(elem):
            is_tensor = isinstance(elem, torch.Tensor)
            if is_tensor and not elem.is_nested:
                return nice_shape(elem.shape)
            elif isinstance(elem, list) or (is_tensor and elem.is_nested):
                shape = (len(elem) if not is_tensor else elem.size(0), )
                inner_shape = None
                for list_elem in elem:
                    inner_shape = inner_shape or list_elem.shape
                    inner_shape = tuple(map(max, inner_shape, list_elem.shape))
                shape = shape + inner_shape
                return nice_shape(shape)
            else:
                raise ValueError(f"Unknown element type {type(elem)}")
        shapes = str({k: shape_for_elem(v) for k, v in self.dataset.items()})
        size = len(self)

        return f"SafetensorsDataset(size={size}, shapes={shapes})"

    @staticmethod
    def _get_len_of_item(i):
        if isinstance(i, torch.Tensor):
            return i.size(0)
        elif isinstance(i, list):
            return len(i)
        raise ValueError(f"{type(i)} is unknown ({i})")

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
            tensor = torch.sparse_coo_tensor(tensor, torch.ones(tensor.size(-1), dtype=dtype), size=tuple(dims))
            tensor = tensor.coalesce()
        else:
            indices = storage[key + ".indices"]
            if key + ".values" not in storage:
                raise ValueError(f"Need {key}.values to restore sparsely stored tensor")
            values = storage[key + ".values"]
            tensor = torch.sparse_coo_tensor(indices, values, size=dims)
            tensor = tensor.coalesce()
        return tensor

    @staticmethod
    def pack_single_tensor(key: str, tensor: torch.Tensor) -> pack_return_t:
        pack: pack_tensor_t
        metadata: pack_metadata_t
        if tensor.is_sparse:
            if tensor.dtype == torch.bool:
                pack = {
                    key: tensor.indices()
                }
            else:
                pack = {
                    key + ".values": tensor.values(),
                    key + ".indices": tensor.indices()
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
            metadata = {
                "nested": True,
                "dtype": repr(tensor.dtype),
                "numel": tensor.size(0),
            }
            return pack, metadata
        return {key: tensor}, None

    def pack_tensor_list(self, key: str, tensors: list[torch.Tensor]) -> pack_return_t:
        if len(tensors) == 0:
            raise ValueError(f"Cannot save an empty list of tensors for key '{key}'")
        if not isinstance(tensors[0], torch.Tensor):
            raise ValueError(f"Elements of '{key}' are no tensors ... element 0 is {type(tensors[0])}")
        if len(tensors) != len(self):
            raise ValueError(f"'{key}' should have {len(self)} elements, but has {len(tensors)}")

        dims = set(map(torch.Tensor.dim, tensors))  # noqa
        if max(dims) > min(dims):
            raise ValueError(f"Elements of '{key}' are of different dimensionality")

        is_sparse = any(map(lambda x: x.is_sparse, tensors))
        is_nested = any(map(lambda x: x.is_nested, tensors))
        if is_sparse and is_nested:
            raise ValueError(f"Cannot pack a mixed list of sparse and nested tensors for key '{key}'")

        if is_sparse:
            sparse_shape = tuple(max(map(lambda x: x.size(dim), tensors)) for dim in range(min(dims)))
            same_size_tensors = list(map(lambda t: torch.sparse_coo_tensor(t.indices(), t.values(), size=sparse_shape), tensors))
            sparse_tensor = torch.stack(same_size_tensors, dim=0).coalesce()
            if sparse_tensor.dtype == torch.bool:
                pack = {key: sparse_tensor.indices()}
            else:
                pack = {key + ".indices": sparse_tensor.indices(), key + ".values": sparse_tensor.values()}
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

    def save_to_file(self, path: Path):
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
            elif isinstance(v, list):
                pack, pack_metadata = self.pack_tensor_list(k, v)

            if pack is not None:
                tensors.update(pack)
                if pack_metadata is not None:
                    metadata[k] = pack_metadata

        metadata = {k: json.dumps(v) for k, v in metadata.items()}
        safetensors.torch.save_file(tensors, path, metadata=metadata)

    @classmethod
    def load_from_file(cls, path: Path):
        metadata = cls._load_safetensors_metadata(path)  # {"size": len}
        tensors = safetensors.torch.load_file(path, device="cpu")
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
            elif meta.get("list", False):
                tensor = cls.unpack_list_tensor(k, metadata, meta, tensors)
            else:
                raise ValueError(f"Cannot unpack stored tensor {k} with metadata = {meta}")
            dataset[k] = tensor
        return SafetensorsDataset(dataset)

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

    @staticmethod
    def _load_safetensors_metadata(fp: str | Path) -> dict[str, Any]:
        with open(fp, 'rb') as f:
            n_bytes = f.read(8)
            n_bytes = int.from_bytes(n_bytes, byteorder='little', signed=False)
            content = f.read(n_bytes)
            content = content.decode("utf-8")
            metadata = json.loads(content)['__metadata__']
            metadata = {k: json.loads(v) for k, v in metadata.items()}
            return metadata
