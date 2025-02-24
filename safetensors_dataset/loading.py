import json
import pathlib
from typing import Union

from .safetensors import SafetensorsDataset
from .safetensors_dict import SafetensorsDict


def load_safetensors(path: Union[str, pathlib.Path]) -> Union[SafetensorsDataset, SafetensorsDict]:
    if isinstance(path, str):
        path = pathlib.Path(path)
    if path.suffixes == [".safetensors", ".index", ".json"]:
        index_path = path
    elif path.with_suffix(".safetensors.index.json").exists():
        index_path = path.with_suffix(".safetensors.index.json")
    else:
        return SafetensorsDataset.load_from_file(path)

    with open(index_path) as f:
        index_dict = json.load(f)

    return SafetensorsDict({
        name: SafetensorsDataset.load_from_file(path.parent / dataset)
        for name, dataset in index_dict.items()
    })
