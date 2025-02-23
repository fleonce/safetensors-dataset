import typing_extensions
from pathlib import Path
from typing import Callable, Optional, Mapping

import torch
from more_itertools.more import first
from typing_extensions import Self

from .safetensors import SafetensorsDataset
from .utils import TensorLayout


class SafetensorsDatasetDict(dict[str, SafetensorsDataset]):
    def __getitem__(self, item: str) -> SafetensorsDataset:
        return super().__getitem__(item)

    @property
    def device(self) -> torch.device:
        return first(map(lambda x: x.device, self.values()))

    def to(self, device: torch.device | int | str) -> Self:
        raise NotImplementedError

    def map(
        self,
        func: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        info: Optional[Mapping[str, TensorLayout]] = None,
        use_tqdm: bool = True,
        batched: bool = False,
        batch_size: int = 1,
    ) -> "SafetensorsDatasetDict":
        raise NotImplementedError

    def select(self, indices: dict[str, list[int]], use_tqdm: bool = False): ...

    def rename(self, key: str, new_key: str): ...

    def info(self) -> Mapping[str, TensorLayout]: ...

    def save_to_file(self, path: Path): ...

    @classmethod
    def load_from_file(cls, path: Path) -> typing_extensions.Self: ...


