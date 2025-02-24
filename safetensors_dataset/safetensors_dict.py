import json
import typing_extensions
from pathlib import Path
from typing import Callable, Optional, Mapping, Union

import torch
from more_itertools.more import first

from .safetensors import SafetensorsDataset
from .utils import TensorLayout


class SafetensorsDict(dict[str, SafetensorsDataset]):
    def __getitem__(self, item: str) -> SafetensorsDataset:
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
        if index_path.suffixes != [".safetensors", ".index", ".json"]:
            index_path = index_path.with_suffix(".safetensors.index.json")
        index_dict = {
            name: path.with_stem(path.stem + "_" + name)
            for name, dataset in self.items()
        }
        for name, dataset in self.items():
            dataset_path = index_dict.get(name)
            dataset.save_to_file(dataset_path)

        with open(index_path, "w") as f:
            json.dump({key: value.name for key, value in index_dict.items()}, f, indent=2)


    @classmethod
    def load_from_file(cls, path: Path) -> typing_extensions.Self: ...


