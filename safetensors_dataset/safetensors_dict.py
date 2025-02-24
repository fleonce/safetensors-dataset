import json
import typing_extensions
from pathlib import Path
from typing import Callable, Optional, Mapping, Union, TypeAlias

import torch
from more_itertools.more import first

from .safetensors import SafetensorsDataset
from .utils import TensorLayout

STK: TypeAlias = Union[str, int]

class SafetensorsDict(dict[STK, SafetensorsDataset]):
    def __getitem__(self, item: STK) -> SafetensorsDataset:
        return super().__getitem__(item)

    @property
    def device(self) -> torch.device:
        return first(map(lambda x: x.device, self.values()))

    def to(self, device: torch.device | int | str) -> "SafetensorsDict":
        return SafetensorsDict({
            name: dataset.to(device) for name, dataset in self.items()
        })

    def map(
        self,
        func: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        info: Optional[Mapping[str, TensorLayout]] = None,
        use_tqdm: bool = True,
        batched: bool = False,
        batch_size: int = 1,
    ) -> "SafetensorsDict":
        return SafetensorsDict({
            name: dataset.map(
                func,
                info,
                use_tqdm,
                batched,
                batch_size
            )
            for name, dataset in self.items()
        })

    def select(self, indices: dict[str, list[int]] | list[int], use_tqdm: bool = False):
        pass

    def rename(self, key: str, new_key: str): ...

    def info(self) -> Mapping[str, TensorLayout]: ...

    def save_to_file(self, path: Union[str, Path]):
        if not isinstance(path, Path):
            path = Path(path)

        index_path = path
        if index_path.suffix == ".safetensors":
            index_path = index_path.parent / index_path.stem / "index.json"
        else:
            index_path = index_path / "index.json"
        index_dict = {
            name: index_path.parent / (str(name) + ".safetensors")  # path.with_stem(path.stem + "_" + name)
            for name, dataset in self.items()
        }
        if not index_path.parent.exists():
            index_path.parent.mkdir(parents=True, exist_ok=True)
        for name, dataset in self.items():
            dataset_path = index_dict[name]
            dataset.save_to_file(dataset_path)

        with open(index_path, "w") as f:
            json.dump([{"split": key, "file": value.name} for key, value in index_dict.items()], f, indent=2)


    @classmethod
    def load_from_file(cls, path: Path) -> typing_extensions.Self: ...


