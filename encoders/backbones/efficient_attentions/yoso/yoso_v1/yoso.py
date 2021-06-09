
import torch
import torch.nn as nn
import os
import time
import math

import sys
curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

import kernel

def hashing(X, Y, num_hash, hash_len):

    assert len(X.size()) == 3 # [b, s, d]
    assert len(Y.size()) == 3 # [b, s, d]

    rmat = torch.randn(X.size(0), X.size(2), num_hash * hash_len, device = X.device)
    raise_pow = 2 ** torch.arange(hash_len, device = X.device)

    Xp = torch.matmul(X, rmat).reshape(X.size(0), X.size(1), num_hash, hash_len)
    Yp = torch.matmul(Y, rmat).reshape(Y.size(0), Y.size(1), num_hash, hash_len)
    Xb = (Xp > 0).int()
    Yb = (Yp > 0).int()
    Xh = torch.sum(Xb * raise_pow, dim = -1)
    Yh = torch.sum(Yb * raise_pow, dim = -1)
    return Xh.int(), Yh.int()


class YOSOEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hash_code_len = config["hash_code_len"]
        self.lsh_config = {"hash_code_len":self.hash_code_len}

        self.use_conv = "conv_window" in config
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = config["num_head"], out_channels = config["num_head"],
                kernel_size = (config["conv_window"], 1), padding = (config["conv_window"] // 2, 0),
                bias = False,
                groups = config["num_head"])

    def forward(self, Q, K, V, mask):

        if self.use_conv:
            conv_V = self.conv(V * mask[:, None, :, None])

        mask = mask.int()

        batch_size, num_heads, seq_len, head_dim = Q.size()

        Q = Q.reshape(batch_size * num_heads, seq_len, head_dim)
        K = K.reshape(batch_size * num_heads, seq_len, head_dim)
        V = V.reshape(batch_size * num_heads, seq_len, head_dim)

        mask = mask[:, None, :].repeat(1, num_heads, 1).reshape(batch_size * num_heads, seq_len)

        Q, K = kernel.normalize([Q, K])
        X = kernel.Cumulation.apply(mask, mask, Q, K, V, self.lsh_config)

        X = kernel.normalize(X)

        X = X.reshape(batch_size, num_heads, seq_len, head_dim)

        if self.use_conv:
            X += conv_V

        return X

    def extra_repr(self):
        return f'hash_code_len={self.hash_code_len}'

class YOSOAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hash_code_len = config["hash_code_len"]
        self.use_fast_hash = config["use_fast_hash"]
        self.num_hash = config["num_hash"]
        self.n2_backward = config["n2_backward"]

        self.lsh_config = {
            "hash_code_len":self.hash_code_len,
            "use_fast_hash":self.use_fast_hash,
            "num_hash":self.num_hash,
            "n2_backward":self.n2_backward
        }

        self.use_conv = "conv_window" in config
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = config["num_head"], out_channels = config["num_head"],
                kernel_size = (config["conv_window"], 1), padding = (config["conv_window"] // 2, 0),
                bias = False,
                groups = config["num_head"])

    def forward(self, Q, K, V, mask):

        if self.use_conv:
            conv_V = self.conv(V * mask[:, None, :, None])

        mask = mask.int()

        batch_size, num_heads, seq_len, head_dim = Q.size()

        Q = Q.reshape(batch_size * num_heads, seq_len, head_dim)
        K = K.reshape(batch_size * num_heads, seq_len, head_dim)
        V = V.reshape(batch_size * num_heads, seq_len, head_dim)

        mask = mask[:, None, :].repeat(1, num_heads, 1).reshape(batch_size * num_heads, seq_len)

        if head_dim < 32:
            Q = torch.cat([Q, torch.zeros(batch_size * num_heads, seq_len, 32 - head_dim, device = Q.device)], dim = -1)
            K = torch.cat([K, torch.zeros(batch_size * num_heads, seq_len, 32 - head_dim, device = K.device)], dim = -1)
            V = torch.cat([V, torch.zeros(batch_size * num_heads, seq_len, 32 - head_dim, device = V.device)], dim = -1)

        if self.training:
            Q, K = kernel.normalize([Q, K])

        X = kernel.LSHCumulation.apply(mask, mask, Q, K, V, self.lsh_config)

        if head_dim < 32:
            X = X[:, :, :head_dim]

        X = kernel.normalize(X)

        X = X.reshape(batch_size, num_heads, seq_len, head_dim)

        if self.use_conv:
            X += conv_V

        return X

    def extra_repr(self):
        return f'hash_code_len={self.hash_code_len}, num_hash={self.num_hash}, fast_hash={self.use_fast_hash}, n2_backward={self.n2_backward}'
