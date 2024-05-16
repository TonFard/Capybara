import importlib.metadata
import importlib.util


def is_torch_bf16_gpu_available():
    import torch
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def is_torch_bf16_available_on_device(device):
    import torch
    if device == "cuda":
        return is_torch_bf16_gpu_available()
    try:
        x = torch.zeros(2, 2, dtype=torch.bfloat16).to(device)
        _ = x @ x
    except:  # noqa: E722
        # TODO: more precise exception matching, if possible.
        # most backends should return `RuntimeError` however this is not guaranteed.
        return False

    return True
import flash_attn
import importlib

def is_flash_attn_2_available():

    if not _is_package_available("flash_attn"):
        return False

    # Let's add an extra check to see if cuda is available
    import torch

    if not torch.cuda.is_available():
        return False

    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    
    elif torch.version.hip:
        # TODO: Bump the requirement to 2.1.0 once released in https://github.com/ROCmSoftwarePlatform/flash-attention
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.0.4")
    else:
        return False

