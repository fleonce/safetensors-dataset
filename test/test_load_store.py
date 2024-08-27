import os
from pathlib import Path
from unittest import TestCase

import torch

from safetensors_dataset import SafetensorsDataset


def try_delete_file(path: Path):
    if path.exists():
        os.remove(path)


def check_dtypes(*dtypes: torch.dtype):
    def inner(func):
        def wrapper(*args, **kwargs):
            for dtype in dtypes:
                func(*args, **kwargs, dtype=dtype)
        return wrapper
    return inner


class StoreDatasetTestCase(TestCase):

    @staticmethod
    def store_and_reload_dataset(dataset: SafetensorsDataset):
        save_path = Path.cwd() / "dataset.safetensors"
        try:
            dataset.save_to_file(save_path)
            dataset = dataset.load_from_file(save_path)
        finally:
            try_delete_file(save_path)
        return dataset

    def check_datasets_are_equal(self, dataset: SafetensorsDataset, comparison: SafetensorsDataset):
        self.assertEqual(dataset.keys(), comparison.keys())
        self.assertEqual(len(dataset), len(comparison))
        for key in dataset.keys():
            self.check_tensors_are_equal(dataset[key], comparison[key])

    def check_tensors_are_equal(self, tensor: torch.Tensor, comparison: torch.Tensor):
        self.assertEqual(isinstance(tensor, list), isinstance(comparison, list))
        if isinstance(tensor, list):
            self.assertEqual(len(tensor), len(comparison))
            for elem, compare in zip(tensor, comparison):
                self.check_tensors_are_equal(elem, compare)
            return None
        self.assertEqual(tensor.is_nested, comparison.is_nested)
        self.assertEqual(tensor.is_sparse, comparison.is_sparse)
        if tensor.is_nested:
            self.assertTrue(tensor.values().equal(comparison.values()))
        elif tensor.is_sparse:
            self.assertTrue(tensor.values().equal(comparison.values()))
            self.assertTrue(tensor.indices().equal(comparison.indices()))
        else:
            self.assertTrue(tensor.equal(comparison))

    @check_dtypes(torch.float, torch.bfloat16, torch.double)
    def test_store_dataset(self, dtype: torch.dtype):
        dataset = SafetensorsDataset.from_dict(
            {
                "test": torch.randn((32, 128), dtype=dtype)
            }
        )
        loaded_dataset = self.store_and_reload_dataset(dataset)
        self.check_datasets_are_equal(dataset, loaded_dataset)

    @check_dtypes(torch.bool, torch.int, torch.float)
    def test_store_sparse_bool_dataset(self, dtype: torch.dtype):
        dataset = {
            "inputs": torch.randint(10, (32, 128)).eq(0).to_sparse().to(dtype)
        }
        dataset = SafetensorsDataset.from_dict(dataset)
        loaded_dataset = self.store_and_reload_dataset(dataset)
        self.check_datasets_are_equal(dataset, loaded_dataset)

    @check_dtypes(torch.float, torch.bfloat16, torch.double)
    def test_store_nested_dataset(self, dtype: torch.dtype):
        lengths = range(10)
        tensors = [torch.randn(length, dtype=dtype) for length in lengths]
        dataset = SafetensorsDataset.from_dict({
            "values": torch.nested.nested_tensor(tensors)
        })
        loaded_dataset = self.store_and_reload_dataset(dataset)
        self.check_datasets_are_equal(dataset, loaded_dataset)

    @check_dtypes(torch.float, torch.bfloat16)
    def test_store_list_dataset(self, dtype: torch.dtype):
        lengths = range(10)
        tensors = [torch.randn(length, dtype=dtype) for length in lengths]
        dataset = SafetensorsDataset.from_dict({"values": tensors})
        loaded_dataset = self.store_and_reload_dataset(dataset)
        self.check_datasets_are_equal(dataset.pack(), loaded_dataset)
