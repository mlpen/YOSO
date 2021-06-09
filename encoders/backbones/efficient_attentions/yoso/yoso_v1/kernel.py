
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math

src_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cuda")
append_root = lambda files : [os.path.join(src_folder, file) for file in files]
src_files = append_root(
    ['fast_lsh_cumulation_torch.cpp',
     'fast_lsh_cumulation.cu',
     'fast_lsh_cumulation_cuda.cu'])
fast_lsh_cumulation = load('fast_lsh_cumulation', src_files, verbose = True)
print(src_files)
print(fast_lsh_cumulation)

import fast_lsh_cumulation as lsh_cumulation

def lsh_cumulation_v1(Q_mask, Q_hash_code, K_mask, K_hash_code, V, hashtable_capacity):
    return lsh_cumulation.lsh_cumulation(Q_mask, Q_hash_code, K_mask, K_hash_code, V, hashtable_capacity, True, 1)

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

def to_contiguous(inp):
    if type(inp) is list:
        out = []
        for tensor in inp:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            out.append(tensor)
        return out
    else:
        if not inp.is_contiguous():
            inp = inp.contiguous()
        return inp

def normalize(inp):
    if type(inp) is list:
        out = []
        for tensor in inp:
            out.append(nn.functional.normalize(tensor, p = 2, dim = -1))
        return out
    else:
        return nn.functional.normalize(inp, p = 2, dim = -1)

class Cumulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q_mask, K_mask, Q, K, V, config):

        hash_code_len = config["hash_code_len"]
        hashtable_capacity = int(2 ** hash_code_len)

        expectation = (1 - torch.acos(torch.matmul(Q, K.transpose(-1, -2))) / math.pi) ** hash_code_len
        expectation = expectation * Q_mask[:, :, None] * K_mask[:, None, :]
        cumulation_V = torch.matmul(expectation, V)

        ctx.save_for_backward(Q_mask, K_mask, expectation, Q, K, V)
        ctx.config = config

        return cumulation_V

    @staticmethod
    def backward(ctx, grad):

        grad = to_contiguous(grad)

        Q_mask, K_mask, expectation, Q, K, V = ctx.saved_tensors
        config = ctx.config

        hash_code_len = config["hash_code_len"]
        hashtable_capacity = int(2 ** hash_code_len)
        V_dim = grad.size(-1)

        weighted_exp = torch.matmul(grad, V.transpose(-1, -2)) * expectation
        grad_Q = torch.matmul(weighted_exp, (hash_code_len / 2) * K)
        grad_K = torch.matmul(weighted_exp.transpose(-1, -2), (hash_code_len / 2) * Q)
        grad_V = torch.matmul(expectation.transpose(-1, -2), grad)

        return None, None, grad_Q, grad_K, grad_V, None

class LSHCumulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q_mask, K_mask, Q, K, V, config):

        assert Q_mask.size(0) == K_mask.size(0)
        assert Q_mask.size(0) == Q.size(0)
        assert Q_mask.size(0) == K.size(0)
        assert Q_mask.size(0) == V.size(0)
        assert K.size(1) == V.size(1)
        assert Q.size(2) == K.size(2)

        Q_mask, K_mask, Q, K, V = to_contiguous([Q_mask, K_mask, Q, K, V])

        use_cuda = Q_mask.is_cuda
        num_hash = config["num_hash"]
        hash_code_len = config["hash_code_len"]
        hashtable_capacity = int(2 ** hash_code_len)

        if config["use_fast_hash"]:
            Q_hash_code, K_hash_code = lsh_cumulation.fast_hash(Q_mask, Q, K_mask, K, num_hash, hash_code_len, use_cuda, 1)
        else:
            Q_hash_code, K_hash_code = hashing(Q, K, num_hash, hash_code_len)

        cumulation_V = lsh_cumulation.lsh_cumulation(Q_mask, Q_hash_code, K_mask, K_hash_code, V, hashtable_capacity, use_cuda, 1)

        ctx.save_for_backward(Q_mask, K_mask, Q_hash_code, K_hash_code, Q, K, V)
        ctx.config = config

        return cumulation_V

    @staticmethod
    def backward(ctx, grad):

        grad = to_contiguous(grad)

        Q_mask, K_mask, Q_hash_code, K_hash_code, Q, K, V = ctx.saved_tensors
        config = ctx.config

        use_cuda = grad.is_cuda
        hash_code_len = config["hash_code_len"]
        hashtable_capacity = int(2 ** hash_code_len)
        V_dim = grad.size(-1)

        if config["n2_backward"]:
            expectation = (1 - torch.acos(torch.matmul(Q, K.transpose(-1, -2))) / math.pi) ** hash_code_len
            expectation = expectation * Q_mask[:, :, None] * K_mask[:, None, :]
            weighted_exp = torch.matmul(grad, V.transpose(-1, -2)) * expectation
            grad_Q = torch.matmul(weighted_exp, (hash_code_len / 2) * K)
            grad_K = torch.matmul(weighted_exp.transpose(-1, -2), (hash_code_len / 2) * Q)
            grad_V = torch.matmul(expectation.transpose(-1, -2), grad)
        else:
            grad_V = lsh_cumulation.lsh_cumulation(K_mask, K_hash_code, Q_mask, Q_hash_code, grad, hashtable_capacity, use_cuda, 1)
            grad_Q = lsh_cumulation.lsh_weighted_cumulation(
                Q_mask, Q_hash_code, grad, K_mask, K_hash_code, V, (hash_code_len / 2) * K, hashtable_capacity, use_cuda, 4)
            grad_K = lsh_cumulation.lsh_weighted_cumulation(
                K_mask, K_hash_code, V, Q_mask, Q_hash_code, grad, (hash_code_len / 2) * Q, hashtable_capacity, use_cuda, 4)

        return None, None, grad_Q, grad_K, grad_V, None
