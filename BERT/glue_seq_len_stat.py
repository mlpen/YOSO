from dataset import DatasetProcessor, BertPreTrainDatasetWrapper, BertDownsteamDatasetWrapper

import argparse
import torch
import torch.nn as nn
import sys
import time
import os
import math
import json
import copy
import pickle
import numpy as np
import random
import datetime
from collections import OrderedDict
from multiprocessing import Pool
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--task", type = str, help = "downstream task", dest = "task", required = True)
args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(curr_path, "datasets", "GLUE", 'task_config.json'), 'r') as f:
    all_downsteam_task_config = json.load(f)

dataset_config = {
    "vocab_size": 30000,
    "num_workers": 16,
    "files_per_batch": 512,
    "max_seq_len": 512,
    "drop_inst_prob": 0.9,
    "short_seq_prob": 0.1,
    "max_mask_token": 80,
    "max_mask_ratio": 0.15,
    "mask_token_prob": {"mask": 0.8, "original": 0.1, "random": 0.1}
}

downsteam_task = args.task
downsteam_task_config = all_downsteam_task_config["task"][downsteam_task]
downsteam_task_config["task"] = downsteam_task
downsteam_task_config["batch_size"] = 128
downsteam_task_config["file_path"] = os.path.join(curr_path, "datasets", "GLUE", "nlp_benchmarks.pickle")

dataset_root_folder = os.path.join(curr_path, "datasets", "ALBERT-pretrain")
data = DatasetProcessor(dataset_root_folder, dataset_config)

train = downsteam_task_config["train"]
train_downsteam_data = BertDownsteamDatasetWrapper(data, downsteam_task_config["file_path"], downsteam_task_config["task"], train)
train_downsteam_dataloader = torch.utils.data.DataLoader(train_downsteam_data, batch_size = downsteam_task_config["batch_size"], pin_memory = True)
train_downsteam_dataloader_iter = iter(train_downsteam_dataloader)

seq_lens = []
for inputs in train_downsteam_dataloader_iter:
    seq_lens.extend(inputs["sentence_mask"].sum(dim = -1).tolist())
    
for val in downsteam_task_config["val"]:
    val_downsteam_data = BertDownsteamDatasetWrapper(data, downsteam_task_config["file_path"], downsteam_task_config["task"], val)
    val_downsteam_dataloader = torch.utils.data.DataLoader(val_downsteam_data, batch_size = 128, pin_memory = True)
    val_downsteam_dataloader_iter = iter(val_downsteam_dataloader)
    
    for inputs in val_downsteam_dataloader_iter:
        seq_lens.extend(inputs["sentence_mask"].sum(dim = -1).tolist())

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(seq_lens, bins = 60)
ax.set_title(f"{downsteam_task}")
ax.set_xlabel(f"SEQ_LEN  --  99 percentile: {int(np.percentile(seq_lens, 99))}, 95 percentile: {int(np.percentile(seq_lens, 95))}, 90 percentile: {int(np.percentile(seq_lens, 90))}, mean: {int(np.mean(seq_lens))}")
fig.savefig(os.path.join(curr_path, "glue-stat", f"glue-seq_len-{downsteam_task}.png"), dpi = 400)

