import json
import re
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
import torch.utils.data


class SafetensorsDataset(torch.utils.data.Dataset):
    dataset: dict[str, list[torch.Tensor] | torch.Tensor]
    layout: dict[str, bool]

    def __init__(self, dataset=None):
        self.dataset = dataset or {}

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.dataset.items()}

    def __len__(self):
        return next((self._get_len_of_item(v) for v in self.dataset.values()), 0)

    @staticmethod
    def _get_len_of_item(i):
        if isinstance(i, torch.Tensor):
            return i.size(0)
        elif isinstance(i, list):
            return len(i)
        raise ValueError(f"{type(i)} is unknown ({i})")

    def __getitems__(self, indices: list[int, ...]):
        return {k: self._get_items_from_tensor(v, indices) for k, v in self.dataset.items()}

    @staticmethod
    def _get_items_from_tensor(t: torch.Tensor, indices: list[int, ...]):
        if isinstance(t, list) or t.is_nested or t.is_sparse:
            return [t[i] for i in indices]
        return t[indices]

    _dtype_to_str = {
        torch.bool: "bool",
        torch.long: "long",
    }
    _str_to_dtype = {"long": torch.long, "bool": torch.bool}

    def save_to_file(self, path: Path):
        metadata = {"size": len(self)}
        tensors = OrderedDict()
        for k, v in self.dataset.items():
            if isinstance(v, torch.Tensor) and not v.is_sparse:
                tensors[k] = v
            elif isinstance(v, torch.Tensor) and v.is_sparse:
                raise ValueError()
            elif isinstance(v, list):
                assert len(v) > 0
                assert isinstance(v[0], torch.Tensor)
                assert len(v) == metadata.get("size"), \
                    f"Length of values for '{k}' ({len(v)}) does not match dataset length {metadata.get('size')}"
                if v[0].is_sparse and v[0].dtype in self._dtype_to_str:
                    v_zero: torch.Tensor = v[0]
                    max_dim = [max([vs.size(i) for vs in v]) for i in range(v_zero.dim())]
                    metadata[k] = {"sparse": True, "dims": max_dim, "dtype": self._dtype_to_str.get(v[0].dtype)}

                    def list_of_sparse_tensors_to_uniform_size(lis):
                        return [torch.sparse_coo_tensor(vs.indices(), vs.values(), size=max_dim) for vs in lis]

                    vvs = list_of_sparse_tensors_to_uniform_size(v)

                    def stack_list_of_sparse_tensors(lis) -> torch.Tensor:
                        return torch.stack(lis, dim=0).coalesce()

                    vvs = stack_list_of_sparse_tensors(vvs)
                    if vvs.dtype == torch.bool:
                        vvs = vvs.indices()
                        tensors[k] = vvs
                    else:
                        tensors[f"{k}.indices"] = vvs.indices()
                        tensors[f"{k}.values"] = vvs.values()
                    continue
                elif v[0].is_sparse and v[0].dtype not in self._dtype_to_str:
                    raise ValueError(f"Dtype {v[0].dtype} is unsupported for sparse tensors (Key = {k})")

                if hasattr(torch, "_nested_view_from_buffer"):
                    nested_tensor = torch.nested.as_nested_tensor(v)
                    assert all(i.dim() == 1 for i in v), (f"Unsupported number of dimensions for nested tensor "
                                                          f"'{k}'")
                    buffer = torch.cat(v, dim=-1)
                    tensors[f"{k}.buffer"] = buffer
                    tensors[f"{k}.sizes"] = nested_tensor._nested_tensor_size()
                    tensors[f"{k}.strides"] = nested_tensor._nested_tensor_strides()
                    tensors[f"{k}.storage_offsets"] = nested_tensor._nested_tensor_storage_offsets()
                    metadata[k] = {"nested": True}
                else:
                    warnings.warn(f"Storing lists of (small) tensors is slow with safetensors, a recent PyTorch "
                                    f"nightly is required to handle this case efficiently. Install via. "
                                    f"'pip3 install --pre torch "
                                    f"--index-url https://download.pytorch.org/whl/nightly/cu118'")
                    for i, t in enumerate(v):
                        tensors[f"{k}.{i}"] = t

        metadata = {k: json.dumps(v) for k, v in metadata.items()}
        safetensors.torch.save_file(tensors, path, metadata=metadata)

    @classmethod
    def load_from_file(cls, path: Path):
        metadata = cls.load_safetensors_metadata(path)  # {"size": len}
        tensors = safetensors.torch.load_file(path, device="cpu")
        dataset = {}
        keys = set()
        for k in tensors.keys():
            if not re.search(r"\.[0-9]+$", k):  # not endswith(number)
                info = k.split('.')
                if len(info) > 1:
                    keys.add('.'.join(info[:-1]))
                else:
                    keys.add(k)
            else:
                match = re.search(r"\.[0-9]+$", k)
                keys.add(k[:match.start()])

        for k in keys:
            if k not in metadata:
                size = metadata.get("size")
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
