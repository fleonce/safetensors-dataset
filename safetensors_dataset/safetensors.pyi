from pathlib import Path
from typing import Callable, overload

import torch.utils.data
from torch import Tensor

class SafetensorsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None):
        pass

    def filter(self, filter_fn: Callable[[dict[str, Tensor]], bool], tqdm: bool = True) -> "SafetensorsDataset":
        pass

    def keys(self) -> set[str]:
        pass

    @overload
    def __getitem__(self, item: str) -> Tensor:
        pass
    @overload
    def __getitem__(self, item: int) -> dict[str, Tensor]:
        pass

    def __getitems__(self, items: list[int, ...]) -> list[dict[str, Tensor], ...]:
        pass

    def __len__(self) -> int:
        pass

    def save_to_file(self, path: Path):
        pass

    @classmethod
    def load_from_file(cls, path: Path) -> SafetensorsDataset:
        pass

    @classmethod
    def from_dict(cls, x: dict[str, Tensor]) -> SafetensorsDataset:
        pass

    @classmethod
    def from_list(cls, x: list[dict[str, Tensor]]) -> SafetensorsDataset:
        pass

