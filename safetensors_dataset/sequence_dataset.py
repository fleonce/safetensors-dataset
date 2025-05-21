import inspect
import warnings
from collections import deque
from typing import (
    Iterable,
    Mapping,
    Any,
    Optional,
    Callable,
    overload,
    Literal,
    Sized,
    MutableMapping,
    Sequence,
    Union,
)

import more_itertools
import torch
import safetensors.torch
from more_itertools.more import first
from tqdm import tqdm

from safetensors_dataset.dict_dataset import SafetensorsDataset, ShardedSafetensorsDataset
from safetensors_dataset.utils import TensorLayout, _map_batch_into_dataset, _apply_function_to_iterable


class CachingIterable:
    def __init__(self, generator):
        self.generator = iter(generator)
        self.cache = deque()
        self.size = 0
        self.exhausted = False

    def __len__(self):
        if not self.exhausted:
            return sum(map(lambda _: 1, self))
        return self.size

    def __next__(self):
        try:
            entry = next(self.generator)
            self.cache.append(entry)
            self.size += 1
            return entry
        except StopIteration:
            self.exhausted = True
            self.generator = None
            raise

    def __iter__(self):
        if self.exhausted:
            return iter(self.cache)
        return self


class SequenceSafetensorsDataset:
    @staticmethod
    def from_iterable(iterable: Iterable[Mapping[str, Any]]):
        return SequenceSafetensorsDataset(iterable)

    dataset: Iterable[Mapping[str, Any]]

    def __init__(self, dataset: Optional[Iterable[Mapping[str, Any]]] = None):
        if dataset is None:
            dataset = tuple()
        elif not isinstance(dataset, Sequence):
            dataset = CachingIterable(dataset)

        self.dataset = dataset

    def __contains__(self, item: str):
        # we assume a homogenous dataset,
        # so we can just check the first entry
        # and test if it contains the key
        try:
            return item in first(self.dataset)
        except ValueError:
            # empty dataset
            return False

    def shard(
        self,
        chunk_size: int = 5000,
        preprocess_if_unprocessed: bool = True,
    ) -> "ShardedSafetensorsDataset | SafetensorsDataset":
        if len(self) <= chunk_size:
            raise ValueError(f"Dataset size is smaller than chunk size ({len(self)} < {chunk_size})")

        length = len(self)
        num_chunks, remainder = divmod(length, chunk_size)
        num_chunks = num_chunks + (remainder != 0)

        chunk_datasets: tuple[dict[str, Union[torch.Tensor, list[Any]]], ...] = tuple(
            dict()
            for _ in range(num_chunks)
        )

        for pos, chunk in enumerate(more_itertools.batched(self.dataset, n=chunk_size, strict=False)):
            for entry in chunk:
                for key, value in entry.items():
                    if key not in chunk_datasets[pos]:
                        chunk_datasets[pos][key] = list()
                    chunk_datasets[pos][key].append(value)

        return ShardedSafetensorsDataset(
            tuple(
                SafetensorsDataset(chunk, preprocess=preprocess_if_unprocessed)
                for chunk in chunk_datasets
            )
        )

    @overload
    def filter(
        self,
        filter_fn: Callable[[dict[str, Any]], bool],
        strict: Literal[False] = False,
        use_tqdm: bool = True,
    ):
        ...

    @overload
    def filter(
        self,
        filter_fn: Callable[[Mapping[str, torch.Tensor]], bool],
        strict: Literal[True] = True,
        use_tqdm: bool = True,
    ):
        ...


    def filter(
        self,
        filter_fn: Callable[[Mapping[str, Any]], bool],
        strict: bool = False,
        use_tqdm: bool = True,
    ):
        if strict:
            def strict_filter_fn(x: Mapping[str, Any]):
                for key, value in x.items():
                    if not isinstance(value, torch.Tensor):
                        raise ValueError(f"{key} must be a tensor, got {type(value)}")
                return filter_fn(x)
            filter_fn = strict_filter_fn

        return SequenceSafetensorsDataset(
            element for element
            in tqdm(self.dataset, leave=False, disable=not use_tqdm)
            if filter_fn(element)
        )

    def keys(self) -> set[str]:
        try:
            return set(first(self.dataset).keys())
        except ValueError:
            return set()

    def __getitem__(self, item: str) -> Iterable[Any]:
        if isinstance(item, str):
            return (element[item] for element in self.dataset)
        raise ValueError(type(item))

    def __len__(self):
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        return sum(map(lambda _: 1, self.dataset))

    @property
    def device(self) -> torch.device:
        try:
            first_elem = first(self.dataset)
            return first(filter(lambda x: isinstance(x, torch.Tensor), first_elem.values())).device
        except ValueError:
            raise ValueError("Cannot determine device in empty dataset")

    def to(self, device: torch.device | str | int) -> "SequenceSafetensorsDataset":
        self_device = self.device
        if self_device == device:
            return self

        def _move_to_device(entry: Mapping[str, Any]):
            return {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in entry.items()
            }

        return self.__class__(
            _move_to_device(element) for element in self.dataset
        )

    @overload
    def map(
        self,
        func: Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
        info: Optional[Mapping[str, TensorLayout]] = None,
        strict: Literal[True] = True,
        use_tqdm: bool = True,
        batched: bool = False,
        batch_size: int = 1,
    ) -> "SafetensorsDataset":
        pass

    def map(
        self,
        func: Callable[[Mapping[str, Any]], Mapping[str, Any] | Sequence[Mapping[str, Any]]],
        info: Optional[Mapping[str, TensorLayout]] = None,
        strict: bool = False,
        use_tqdm: bool = True,
        batched: bool = False,
        batch_size: int = 1,
    ) -> "SafetensorsDataset":
        def batch_fn():
            for batch in more_itertools.batched(self.dataset, n=batch_size):
                out_batch = {_key: [] for _key in batch[0].keys()}
                for _element in batch:
                    for _key, _value in out_batch.items():
                        _value.append(_element[_key])
                yield out_batch

        items = (
            self.dataset
            if not batched
            else batch_fn()
        )

        dataset = _apply_function_to_iterable(func, items, len(self), batched, batch_size, disable_tqdm=not use_tqdm)

        return SafetensorsDataset(dataset, preprocess=True)

    def select(self, indices: list[int], use_tqdm=False) -> "SafetensorsDataset":
        raise NotImplementedError()

    def info(self) -> Mapping[str, TensorLayout]:
        return {
            key: TensorLayout.VARYING_DIM_SIZE
            for key in self.keys()
        }

    def rename(self, key: str, new_key: str):
        raise NotImplementedError

    def __repr__(self):
        def nice_shape(shape):
            return "[" + " x ".join(map(str, shape)) + "]"

        def shape_for_elem(elem):
            is_tensor = isinstance(elem, torch.Tensor)
            if is_tensor and not elem.is_nested:
                return nice_shape(elem.shape)
            elif isinstance(elem, Sequence) or (is_tensor and elem.is_nested):
                shape = (len(elem) if not is_tensor else elem.size(0),)
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

        return f"SequenceSafetensorsDataset(size={size})"

    def __del__(self):
        del self.dataset


SequenceSafetensorsDataset.from_iterable(a for a in range(100))