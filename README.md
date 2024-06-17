# safetensors-dataset

`safetensors-dataset` is a very simple and tiny package adding support to efficiently load and store Pytorch datasets in the safetensors file format:

```python3
import torch
from safetensors_dataset import SafetensorsDataset

dataset = SafetensorsDataset.from_dict({
    "x": torch.randn(8, 12, 3),
    "y": torch.randint(12, (8,))
})
dataset.save_to_file("test.safetensors")
```
