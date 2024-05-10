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


class CapybaraRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float=1e-6):
        """
        CapybaraRMSNorm
        """
        super(CapybaraRMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.tp(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class CapybaraRotaryEmbedding():
    raise NotImplementedError("Reference DeepSeedv2")


class CapybaraAttention(nn.Module):
    def __init__(self, ):
        super(CapybaraAttention, self).__init__()


class CapybaraMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = None,
        bias: bool = False,
        drop_prob: float = 0.0
    ):
        """
        CapybaraMLP reference nanoGPT MLP and DeepSeekv2 MLP
        """
        super(CapybaraMLP, self).__init__()
        if intermediate_size is None:
            intermediate_size = (int(hidden_size * 8/3 / 128) + 1) * 128
        self.in_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.act(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x


class CapybaraDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_hidden_layers: int, ):
        super(CapybaraDecoderLayer, self).__init__()
        self.norm1 = CapybaraRMSNorm(hidden_size)
        self.attn = CapybaraAttention()
        self.norm2 = CapybaraRMSNorm(hidden_size)
        self.mlp = CapybaraMLP(hidden_size)
    
    def forward(self, x):
        pass

class Capybara(nn.Module):
    def __init__(self, ):
        raise NotImplementecdError("It hasn't been implemented yet")


model = MultiHeadLatentAttention()

