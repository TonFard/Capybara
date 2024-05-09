import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union


@dataclass
class ModelConfig:
    dim: int = field(default=1024)  # Dimension of the model
    n_layers: int = field(default=32)  # Number of layers in the transformer
    n_heads: int = field(default=32)  # Number of attention heads
    n_kv_heads: Optional[int] = field(default=None)  # Number of key-value heads (optional, defaults to n_heads)
    vocab_size: int = field(default=50257)  # Vocabulary size
    norm_eps: float = field(default=1e-5)  # Epsilon value for normalization
    max_batch_size: int = field(default=32)  # Maximum batch size for training
    max_seq_len: int = field(default=2048)  # Maximum sequence length
    device: str = field(default=None)  # Device to run the model on (optional)


class MultiHeadLatentAttention():
    raise NotImplementedError("Reference DeepSeedv2 MLA")

class CapybaraRMSNorm():
    def __init__(self, hidden_size, eps=1e-6):
        """
        CapybaraRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.tp(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    


    raise NotImplementedError("Reference llama3")

class RoPE():
    raise NotImplementedError("Reference DeepSeedv2")

class Capybara(nn.Module):
    def __init__(self, ):
        raise NotImplementecdError("It hasn't been implemented yet")

class CapybaraDecoderLayer():
    raise NotImplementedError("It hasn't been implemented yet")
 
class CapybaraMLP():
    raise NotImplementedError("It hasn't been implemented yet")




model = MultiHeadLatentAttention()

