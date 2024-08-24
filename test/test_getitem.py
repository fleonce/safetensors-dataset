from unittest import TestCase

import torch

from safetensors_dataset import SafetensorsDataset


class GetItemTestCase(TestCase):
    dataset: SafetensorsDataset

    def setUp(self):
        self.inputs = torch.randn((32, 16, 16))
        self.dataset = SafetensorsDataset.from_dict({
            "inputs": self.inputs
        })

    def test_getitem_by_name(self):
        self.assertTrue(self.dataset["inputs"].equal(self.inputs))

    def test_getitem_by_index(self):
        for index in range(self.inputs.size(0)):
            elem = self.dataset[index]
            self.assertTrue(elem["inputs"].equal(self.inputs[index]))
