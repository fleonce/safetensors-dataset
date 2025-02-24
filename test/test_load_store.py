import os
import unittest
from pathlib import Path
from unittest import TestCase

import torch

from safetensors_dataset import SafetensorsDataset, SafetensorsDict, load_safetensors


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

    def test_store_dict(self):
        ds = SafetensorsDict(
            {
                "train": SafetensorsDataset({"label": torch.arange(10)}),
                "test": SafetensorsDataset({"label": torch.arange(10) + 10}),
            }
        )

        ds.save_to_file("test.safetensors")

        loaded = load_safetensors("test.safetensors")
        for name, dataset in ds.items():
            self.check_datasets_are_equal(dataset, loaded[name])

    def test_store_int_dict(self):
        ds = SafetensorsDict(
            {
                0: SafetensorsDataset({"label": torch.arange(10)}),
                1: SafetensorsDataset({"label": torch.arange(10) + 10}),
            }
        )

        ds.save_to_file("ints")

        loaded = load_safetensors("ints")
        for name, dataset in ds.items():
            self.check_datasets_are_equal(dataset, loaded[name])

    @staticmethod
    def store_and_reload_dataset(dataset: SafetensorsDataset):
        save_path = Path.cwd() / "dataset.safetensors"
        try:
            dataset.save_to_file(save_path)
            dataset = load_safetensors(save_path)
        finally:
            try_delete_file(save_path)
        return dataset

    def check_datasets_are_equal(self, dataset: SafetensorsDataset, comparison: SafetensorsDataset):
        self.assertEqual(dataset.keys(), comparison.keys())
        self.assertEqual(len(dataset), len(comparison))
        for key in dataset.keys():
            self.check_tensors_are_equal(dataset[key], comparison[key])

    def format_tensor_not_matching(self, tensor: torch.Tensor, comparison: torch.Tensor):
        return f"Expected tensors to match: {tensor} vs {comparison}"

    def check_tensors_are_equal(self, tensor: torch.Tensor, comparison: torch.Tensor):
        self.assertEqual(isinstance(tensor, list), isinstance(comparison, list))
        if isinstance(tensor, list):
            self.assertEqual(len(tensor), len(comparison))
            for elem, compare in zip(tensor, comparison):
                self.check_tensors_are_equal(elem, compare)
            return None
        self.assertEqual(tensor.is_nested, comparison.is_nested)
        self.assertEqual(tensor.is_sparse, comparison.is_sparse)
        self.assertEqual(tensor.numel(), comparison.numel(), self.format_tensor_not_matching(tensor, comparison))
        self.assertEqual(tensor.dim(), comparison.dim(), self.format_tensor_not_matching(tensor, comparison))
        if tensor.is_nested:
            self.assertTrue(tensor.values().equal(comparison.values()))
            self.check_tensors_are_equal(tensor._nested_tensor_size(), comparison._nested_tensor_size())
            self.check_tensors_are_equal(tensor._nested_tensor_strides(), comparison._nested_tensor_strides())
            self.check_tensors_are_equal(tensor._nested_tensor_storage_offsets(), comparison._nested_tensor_storage_offsets())
        elif tensor.is_sparse:
            self.assertTrue(tensor.values().equal(comparison.values()))
            self.assertTrue(tensor.indices().equal(comparison.indices()))
        else:
            self.assertTrue(tensor.equal(comparison), self.format_tensor_not_matching(tensor, comparison))

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
        print(dataset["values"])
        print(dataset["values"]._nested_tensor_size())
        print(dataset["values"]._nested_tensor_strides())
        print(dataset["values"]._nested_tensor_storage_offsets())
        loaded_dataset = self.store_and_reload_dataset(dataset)
        self.check_datasets_are_equal(dataset, loaded_dataset)

    @check_dtypes(torch.float, torch.bfloat16)
    def test_store_list_dataset(self, dtype: torch.dtype):
        lengths = range(10)
        tensors = [torch.randn(length, dtype=dtype) for length in lengths]
        dataset = SafetensorsDataset.from_dict({"values": tensors})
        loaded_dataset = self.store_and_reload_dataset(dataset)
        self.check_datasets_are_equal(dataset.pack(), loaded_dataset)

    @check_dtypes(torch.int, torch.bool)
    def test_store_sparse_dataset(self, dtype: torch.dtype):
        tensors = [torch.randint(2, (137, 10, 10), dtype=dtype).to_sparse() for _ in range(10)]
        dataset = SafetensorsDataset.from_dict({"values": tensors})
        loaded_dataset = self.store_and_reload_dataset(dataset)
        self.check_datasets_are_equal(dataset.pack(), loaded_dataset)

    def test_store_single_elems(self):
        tensors = list(torch.randint(128, (32,)).unbind())
        dataset = SafetensorsDataset.from_dict({"values": tensors})
        loaded_dataset = self.store_and_reload_dataset(dataset)
        self.check_datasets_are_equal(dataset.pack(), loaded_dataset)


if __name__ == "__main__":
    unittest.main()
