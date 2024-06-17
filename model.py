# du 18758813215
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
nn.LayerNorm

@dataclass
class ModelConfig:
    attention_bias: bool = False
    attention_dropout: float = 0.0
    # bos_token_id: int = 1
    # eos_token_id: int = 2
    hidden_size: int = 768
    hidden_act: str = "gelu"
    ffn_bias: bool = False
    intermediate_size: int = (int(hidden_size * 8/3 / 128) + 1) * 128
    num_attention_heads: int = 16
    num_hidden_layers: int = 4
    num_kv_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    vocab_size: int = 64794
    block_size: int = 1024
    
    def __repr__(self):
        attributes = vars(self)
        lines = []
        for attr, value in attributes.items():
            lines.append(f"\n    {attr}: {value}")
        return f"{self.__class__.__name__}({','.join(lines)}\n)"
    

class RMSNorm(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        norm_eps: float=1e-6,
    ):
        """
        CapybaraRMSNorm
        """
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = norm_eps
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class CasualSelfAttention(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False
    ):
        super(CasualSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=attention_bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.q_head_dim = self.hidden_size // self.num_heads

        self.proj_dropout = nn.Dropout(attention_dropout)
        self.softmax_scale = self.q_head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_size = x.size()  # [batch_size, sequence_length, embedding dimension]
        q, k, v = self.wqkv(x).split(self.hidden_size, dim=2)
        q = q.view(bsz, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)  # [bsz, nh, seq_len, hs]
        k = k.view(bsz, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)  # [bsz, nh, seq_len, hs]
        v = v.view(bsz, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)  # [bsz, nh, seq_len, hs]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, hidden_size)
        attn_output = self.proj_dropout(self.proj(attn_output))
        return attn_output


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        act: str = 'gelu',
        bias: bool = False,
    ):
        """
        MLP reference nanoGPT MLP and DeepSeekv2 MLP
        """
        super(MLP, self).__init__()
        if intermediate_size is None:
            intermediate_size = int((int(hidden_size * 8/3 / 128) + 1) * 128)
        self.in_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        if act.lower() == "gelu":
            self.act = nn.GELU() 
        elif act.lower() == "silu":
            self.act = nn.SiLU()
        else:
            NotImplementedError(f"not support {act} !!!")
        self.out_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x = self.act(x)
        x = self.out_proj(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        intermediate_size: Optional[int], 
        num_attention_heads: int, 
        attention_dropout: float, 
        attention_bias: bool,
        norm_eps: float,
        ffn_bias: bool,
        act: str
    ):
        """
            DecoderLayer
        """
        super(DecoderLayer, self).__init__()
        self.norm1 = RMSNorm(hidden_size, norm_eps)
        self.attn = CasualSelfAttention(hidden_size, num_attention_heads, attention_dropout, attention_bias)
        self.norm2 = RMSNorm(hidden_size, norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, act, ffn_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
    
class Capybara(nn.Module):
    def __init__(self, config):
        super(Capybara, self).__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pte = nn.Embedding(config.block_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    config.attention_dropout,
                    config.attention_bias,
                    config.rms_norm_eps,
                    config.ffn_bias,
                    config.hidden_act
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, idx, target=None):
        device = idx.device
        b, t = idx.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)
        pos_emb = self.pte(pos)
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        if target is not None:
            logits = self.fc(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        else:
            logits = self.fc(x[:, [-1], :])
            loss = None
        return logits, loss

    # def crop_block_size(self, block_size):
    #     assert block_size <= self.config.block_size
    #     self.config.blcok_size = block_size
    #     self.wpe.weight = nn.Parameter(self.wpe.weight[:block_size])
    #     for block in self.layers:
    #         if hasattr(block.attn, "bias"):
    #             block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
    
    @torch.no_grad()
    def generate(self, idx, max_new_token, temperature=1.0, top_k=None):
        for _ in range(max_new_token):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
        
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

# 打印模型的每一层及其参数大小
def print_model_parameters(model):
    print("Layer Name & Parameters")
    print("----------------------------")
    total_params = 0
    for name, parameter in model.named_parameters():
        param_size = parameter.size()
        param_count = torch.prod(torch.tensor(param_size)).item()
        total_params += param_count
        print(f"{name:50} | Size: {str(param_size):30} | Count: {str(param_count):20}")
    print("----------------------------")
    print(f"Total Parameters: {total_params} ({total_params / 1000000:.1f} M)")


if __name__ == "__main__":
    config = ModelConfig()
    hidden_size = config.hidden_size
    input_tensor = torch.rand(32, 10, hidden_size)
    model = Capybara(config)
    print_model_parameters(model)

    # print(model(input_tensor).shape)

