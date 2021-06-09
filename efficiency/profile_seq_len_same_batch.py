import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

from LRA.model_wrapper import ModelForSC
import torch
import copy
import time
import pandas as pd
import json
import numpy as np

model_config = {
    "mixed_precision":True,
    "shared_weight":True,
    "embedding_dim":256,
    "dim":256,
    "hidden_dim":1024,
    "head_dim":64,
    "num_head":4,
    "num_layers":6,
    "vocab_size":512,
    "max_seq_len":64,
    "dropout_prob":0.1,
    "pooling_mode":"MEAN",
    "num_classes":10,
}
attn_config = {
    "softmax":{"attn_type":"softmax"},
    "linformer":{"attn_type":"linformer", "linformer_k":256},
    "reformer":{"attn_type":"reformer", "num_hash":2},
    "performer":{"attn_type":"performer", "rp_dim":256, "kernel_type":"relu"},
    "longformer":{"attn_type":"longformer", "window_size":128, "first_token_view":True},
    "nystrom-64-conv33":{"attn_type":"nystrom", "num_landmarks":64, "conv_kernel_size":33},
    "yoso-v1-h32":{"attn_type":"yoso-v1", "use_fast_hash":True,"hash_code_len":9, "num_hash":32, "n2_backward": False},
    "yoso-v1-h16":{"attn_type":"yoso-v1", "use_fast_hash":True,"hash_code_len":9, "num_hash":16, "n2_backward": False},
    "yoso-v1-h32-conv33":{"attn_type":"yoso-v1", "use_fast_hash":True,"hash_code_len":9, "num_hash":32, "n2_backward": False, "conv_window":33},
    "yoso-v1-h16-conv33":{"attn_type":"yoso-v1", "use_fast_hash":True,"hash_code_len":9, "num_hash":16, "n2_backward": False, "conv_window":33},
    # "yoso-v2-h32":{"attn_type":"yoso-v2", "hashcode_len":9, "num_hash_f":32},
    # "yoso-v2-h16":{"attn_type":"yoso-v2", "hashcode_len":9, "num_hash_f":16},
    # "yoso-v2-h32-conv33":{"attn_type":"yoso-v2", "hashcode_len":9, "num_hash_f":32, "conv_window":33},
    # "yoso-v2-h16-conv33":{"attn_type":"yoso-v2", "hashcode_len":9, "num_hash_f":16, "conv_window":33},
}

def func(model, batch_size, seq_len, training):
    if training:
        input_ids = torch.randint(0, 512, (batch_size, seq_len)).long().cuda()
        labels = torch.randint(0, 10, (batch_size, )).long().cuda()
        masks = torch.ones(batch_size, seq_len).float().cuda()
        out = model(input_ids, masks, labels)
        out["loss"].mean().backward()
    else:
        with torch.no_grad():
            input_ids = torch.randint(0, 512, (batch_size, seq_len)).long().cuda()
            labels = torch.randint(0, 10, (batch_size, )).long().cuda()
            masks = torch.ones(batch_size, seq_len).float().cuda()
            out = model(input_ids, masks, labels)

def get_time_memory(config, batch_size, seq_len, training):
    model = ModelForSC(config).cuda()
    func(model, batch_size, seq_len, training)

    time_list = []
    for _ in range(num_iter):
        torch.cuda.synchronize()
        t0 = time.time()
        func(model, batch_size, seq_len, training)
        torch.cuda.synchronize()
        t1 = time.time()
        time_list.append((t1 - t0) / batch_size)

    per_inst_time_avg = np.mean(time_list) * 1000
    per_inst_time_std = np.std(time_list) * 1000
    memory_per_inst = torch.cuda.max_memory_allocated() / batch_size / 1024 / 1024

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"seq_len={seq_len}, batch_size={batch_size}")
    print(f"per_inst_time_avg={per_inst_time_avg}, per_inst_time_std={per_inst_time_std}")
    print(f"memory_per_inst={memory_per_inst}")

    return per_inst_time_avg, per_inst_time_std, memory_per_inst

num_iter = 10

batch_size_dict = {}
for log_seq_len in reversed(range(7, 13)):
    seq_len = int(2 ** log_seq_len)
    batch_size_dict[seq_len] = 1000000

for attn_type in attn_config:
    with open(f"profile_seq_len_max_batch/{attn_type}.json", "r") as f:
        data = json.load(f)
        for key in data:
            batch_size_dict[int(key)] = min(batch_size_dict[int(key)], data[key]["train"]["batch_size"])
print(batch_size_dict)

for attn_type in attn_config:
    print(f"attn_type={attn_type}")
    results = {}
    for log_seq_len in reversed(range(7, 13)):
        seq_len = int(2 ** log_seq_len)

        results[seq_len] = {}

        config = copy.deepcopy(model_config)
        config.update(attn_config[attn_type])
        config["max_seq_len"] = seq_len
        config["hashcode_len"] = log_seq_len
        config["hash_code_len"] = log_seq_len

        batch_size = batch_size_dict[seq_len]

        per_inst_time_avg, per_inst_time_std, memory_per_inst = get_time_memory(config, batch_size, seq_len, True)
        results[seq_len]["train"] = {
            "batch_size":batch_size,
            "per_inst_time_avg (ms)":round(per_inst_time_avg, 3),
            "per_inst_time_std (ms)":round(per_inst_time_std, 3),
            "memory_per_inst (MB)":round(memory_per_inst, 3),
        }

        print(seq_len)
        print(results[seq_len])

    with open(f"profile_seq_len_same_batch/{attn_type}.json", "w") as f:
        json.dump(results, f, indent = 4)
