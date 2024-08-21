import json
import warnings
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping

import safetensors.torch
from tqdm import trange
import torch
import torch.utils.data


pack_tensor_t = dict[str, torch.Tensor]
pack_metadata_t = dict[str, Any] | None
pack_return_t = tuple[pack_tensor_t, pack_metadata_t]
can_pack_nested_tensors_fast = hasattr(torch, "_nested_view_from_buffer")


class SafetensorsDataset(torch.utils.data.Dataset):
    dataset: dict[str, list[torch.Tensor] | torch.Tensor]
    layout: dict[str, bool]

    def __init__(self, dataset=None):
        self.dataset = dataset or {}

    def filter(
        self,
        filter_fn: Callable[[dict[str, torch.Tensor]], bool],
        tqdm: bool = True
    ):
        filtered_dataset = dict({k: list() for k in self.dataset.keys()})
        from_to = range if not tqdm else partial(trange, leave=False)
        for i in from_to(len(self)):
            elem = self[i]
            if filter_fn(elem):
                for k in filtered_dataset.keys():
                    filtered_dataset[k].append(elem[k])
        return SafetensorsDataset(filtered_dataset)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.dataset.items()}

    def __getitems__(self, indices: list[int, ...]):
        elements_per_key = {k: self._get_items_from_tensor(v, indices) for k, v in self.dataset.items()}
        return [{k: elements_per_key[k][i] for k in elements_per_key.keys()} for i in range(len(indices))]

    def __len__(self):
        return next((self._get_len_of_item(v) for v in self.dataset.values()), 0)

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
    def _get_items_from_tensor(t: torch.Tensor, indices: list[int, ...]):
        if isinstance(t, list) or t.is_nested or t.is_sparse:
            return [t[i] for i in indices]
        return t[indices]

    @staticmethod
    def pack_single_tensor(key: str, tensor: torch.Tensor) -> pack_return_t:
        pack: pack_tensor_t
        metadata: pack_metadata_t
        if tensor.is_sparse:
            pack = {
                key + ".values": tensor.values(),
                key + ".indices": tensor.values()
            }
            metadata = {
                "sparse": True,
                "dtype": repr(tensor.dtype),
                "dims": tensor.shape,
                "numel": 1,
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
                "numel": 1,
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

        dims = map(torch.Tensor.dim, tensors)  # noqa
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
                pack = {key: sparse_tensor}
            else:
                pack = {key + ".indices": sparse_tensor.indices(), key + ".values": sparse_tensor.values()}
            metadata = {
                "sparse": True,
                "dims": sparse_tensor.shape,
                "dtype": sparse_tensor.dtype,
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
            if nested_tensor.dim() > 2:
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

        metadata = {"size": len(self), "version": None}
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
        metadata = cls.load_safetensors_metadata(path)  # {"size": len}
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

        size = metadata.get("size")
        for k in keys:
            meta: Mapping[str, Any] = metadata.get(k, dict())
            if not meta:
                # load a single tensor
                v = tensors[k]
            if meta.get("sparse", False) is True:
                v = cls.unpack_sparse_entry(v, meta)
            elif meta.get("")
            if not meta or meta.get("list", False):
                pass
            if k not in metadata:
                v = [None for _ in range(size)]
                for i in range(size):
                    v[i] = tensors.get(f"{k}.{i}")
            else:
                element_metadata = metadata[k]
                if element_metadata.get("sparse", False):
                    dense_dtype = cls._str_to_dtype.get(element_metadata.get("dtype"))
                    dense_size = (metadata.get("size"), *element_metadata.get("dims"))
                    if dense_dtype == torch.bool:
                        v = tensors[k]
                        v = torch.sparse_coo_tensor(v, torch.ones(v.size(-1), dtype=torch.bool), size=dense_size)
                    else:
                        v = tensors[k + ".indices"]
                        if k + ".values" not in tensors:
                            raise ValueError(f"Key '{k}.values' was not saved in '{path.as_posix()}'")
                        vals = tensors[k + ".values"]
                        v = torch.sparse_coo_tensor(v, vals, size=dense_size)
                    v = v.coalesce()
                elif element_metadata.get("nested", False):
                    assert hasattr(torch, "_nested_view_from_buffer"), \
                        (f"To load nested values, a very recent torch nightly is required. Install via "
                         f"'pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118'")
                    buffer = tensors[k + ".buffer"]
                    sizes = tensors[k + ".sizes"]
                    strides = tensors[k + ".strides"]
                    storage_offsets = tensors[k + ".storage_offsets"]
                    v = torch._nested_view_from_buffer(buffer, sizes, strides, storage_offsets)
                else:
                    raise ValueError(f"Dont know how to handle key {k} with metadata {element_metadata}")

            dataset[k] = v
        return SafetensorsDataset(dataset)

    @staticmethod
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

    @classmethod
    def from_dict(cls, x: dict):
        cls._check_input_dict(x)
        return SafetensorsDataset(x)

    @classmethod
    def from_list(cls, x: list[dict]):
        out = {}
        for i in x:
            for k, v in i.items():
                if k not in out:
                    out[k] = [v]
                else:
                    out[k].append(v)
        return cls.from_dict(out)

    @staticmethod
    def load_safetensors_metadata(fp: str | Path) -> dict[str, Any]:
        with open(fp, 'rb') as f:
            n_bytes = f.read(8)
            n_bytes = int.from_bytes(n_bytes, byteorder='little', signed=False)
            content = f.read(n_bytes)
            content = content.decode("utf-8")
            metadata = json.loads(content)['__metadata__']
            metadata = {k: json.loads(v) for k, v in metadata.items()}
            return metadata
