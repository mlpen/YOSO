from transformers.modeling_longformer import LongformerSelfAttention, LongformerConfig
import torch
import torch.nn as nn

class LongformerAttention(LongformerSelfAttention):
    def __init__(self, config, query, key, value):

        self.window_size = config["window_size"]
        self.first_token_view = config["first_token_view"]
        longformer_config = LongformerConfig()
        longformer_config.num_attention_heads = config["num_head"]
        longformer_config.hidden_size = config["dim"]
        longformer_config.attention_window = [config["window_size"]]

        super().__init__(longformer_config, 0)

        self.query.weight = query.weight
        self.query_global.weight = query.weight

        self.key.weight = key.weight
        self.key_global.weight = key.weight

        self.value.weight = value.weight
        self.value_global.weight = value.weight

        self.query.bias = query.bias
        self.query_global.bias = query.bias

        self.key.bias = key.bias
        self.key_global.bias = key.bias

        self.value.bias = value.bias
        self.value_global.bias = value.bias

    def forward(self, X, mask):
        mask = mask - 1
        if self.first_token_view:
            mask[:, 0] = 1
        return super().forward(hidden_states = X, attention_mask = mask[:, None, None, :])[0]

    def extra_repr(self):
        return f'window_size={self.window_size}'
