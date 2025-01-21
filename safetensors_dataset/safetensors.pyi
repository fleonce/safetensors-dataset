from pathlib import Path
from typing import Callable, overload, Self, Mapping, Any, Optional, Generator

import torch.utils.data
from torch import Tensor

from safetensors_dataset.utils import TensorLayout

pack_tensor_t = dict[str, torch.Tensor]
pack_metadata_t = dict[str, Any] | None
pack_return_t = tuple[pack_tensor_t, pack_metadata_t]

class SafetensorsDataset(torch.utils.data.Dataset):
    dataset: dict[str, list[torch.Tensor] | torch.Tensor]

    def __init__(self, dataset=None, preprocess: bool=False):
        pass

    def pack(self) -> Self: ...

    def filter(self, filter_fn: Callable[[dict[str, Tensor]], bool], tqdm: bool = True) -> "SafetensorsDataset": ...

    def keys(self) -> set[str]: ...

    @overload
    def __getitem__(self, item: str) -> Tensor | list[Tensor]: ...
    @overload
    def __getitem__(self, item: int) -> dict[str, Tensor]: ...

    def __getitems__(self, items: list[int, ...]) -> list[dict[str, Tensor], ...]: ...

    def __len__(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    def to(self, device: torch.device | int | str) -> SafetensorsDataset: ...

    def map(
        self,
        func: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        info: Optional[Mapping[str, TensorLayout]] = None,
        use_tqdm: bool = True,
        batched: bool = False,
        batch_size: int = 1,
    ) -> "SafetensorsDataset": ...

    def _transpose(self, batched=False, batch_size=0) -> Generator[Mapping[str, torch.Tensor], None, None]: ...

    def save_to_file(self, path: Path): ...

    @classmethod
    def load_from_file(cls, path: Path) -> SafetensorsDataset: ...

    @classmethod
    def from_dict(cls, x: dict[str, Tensor | list[Tensor]], *, preprocess: bool=False) -> SafetensorsDataset: ...

    @classmethod
    def from_list(cls, x: list[dict[str, Tensor]], *, preprocess: bool=False) -> SafetensorsDataset: ...

    @staticmethod
    def unpack_list_tensor(key: str, metadata: Mapping[str, Any], meta: Mapping[str, Any], storage: Mapping[str, torch.Tensor]): ...

    @staticmethod
    def unpack_nested_tensor(key: str, metadata: Mapping[str, Any], meta: Mapping[str, Any], storage: Mapping[str, torch.Tensor]): ...

    @staticmethod
    def unpack_sparse_tensor(key: str, metadata: Mapping[str, Any], meta: Mapping[str, Any], storage: Mapping[str, torch.Tensor]): ...

    @staticmethod
    def pack_single_tensor(key: str, tensor: torch.Tensor) -> pack_return_t: ...

    def pack_tensor_list(self, key: str, tensors: list[torch.Tensor]) -> pack_return_t: ...