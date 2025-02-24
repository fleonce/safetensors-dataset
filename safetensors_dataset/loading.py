import json
import pathlib
from os import PathLike
from typing import Union

from .safetensors import SafetensorsDataset
from .safetensors_dict import SafetensorsDict


def load_safetensors(path: Union[str, pathlib.Path]) -> Union[SafetensorsDataset, SafetensorsDict]:
    if isinstance(path, str):
        path = pathlib.Path(path)
    if path.is_dir() and (path / "index.json").exists():
        index_path = path / "index.json"
    elif (path.parent / path.stem / "index.json").exists():
        index_path = (path.parent / path.stem / "index.json")
    else:
        return SafetensorsDataset.load_from_file(path)

    with open(index_path) as f:
        index_dict = json.load(f)

    return SafetensorsDict({
        index["split"]: SafetensorsDataset.load_from_file(index_path.parent / index["file"])
        for index in index_dict
    })


def exists_safetensors(path: Union[str, PathLike]):
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if path.is_dir() and (path / "index.json").exists():
        return True
    elif (path.parent / path.stem / "index.json").exists():
        return True
    else:
        return path.exists()