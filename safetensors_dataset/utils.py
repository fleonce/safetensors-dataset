from typing import cast

import torch


def get_torch_dtype_from_str(dtype: str) -> torch.dtype:
    """
    Convert the string representation of a dtype to the corresponding torch.dtype type

    :param dtype: str
    :return: torch.dtype
    """
    dtype_data = dtype.split(".")
    dtype_name = dtype_data[-1]
    return cast(torch.dtype, getattr(torch, dtype_name))
