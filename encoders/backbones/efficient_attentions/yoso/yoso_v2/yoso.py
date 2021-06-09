
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

import kernel

class YOSOAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hashcode_len = config["hashcode_len"]
        self.num_hash_f = config["num_hash_f"]
        self.num_head = config["num_head"]

        self.use_conv = "conv_window" in config
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (config["conv_window"], 1), padding = (config["conv_window"] // 2, 0),
                bias = False,
                groups = self.num_head)

    def forward(self, Q, K, V, mask):
        if self.use_conv:
            conv_V = self.conv(V * mask[:, None, :, None])

        batch_size, num_heads, seq_len, head_dim = Q.size()

        Q = Q.reshape(batch_size * num_heads, seq_len, head_dim)
        K = K.reshape(batch_size * num_heads, seq_len, head_dim)
        V = V.reshape(batch_size * num_heads, seq_len, head_dim)

        mask = mask.int()[:, None, :].repeat(1, num_heads, 1).reshape(batch_size * num_heads, seq_len)

        if self.num_hash_f < 0:
            Q = kernel.normalize(Q)
            K = kernel.normalize(K)
            X = kernel.yoso_e(Q, K, V, mask, mask, self.hashcode_len)
        else:
            if self.training:
                Q = kernel.normalize(Q)
                K = kernel.normalize(K)
            X = kernel.yoso(
                Q.contiguous(), K.contiguous(), V.contiguous(),
                mask.contiguous(), mask.contiguous(),
                self.num_hash_f, self.hashcode_len
            )

        X = kernel.normalize(X)

        X = X.reshape(batch_size, num_heads, seq_len, head_dim)

        if self.use_conv:
            X += conv_V

        return X

    def extra_repr(self):
        return f'num_hash_f={self.num_hash_f}, hashcode_len={self.hashcode_len}, use_conv={self.use_conv}'
