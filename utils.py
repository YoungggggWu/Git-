import math
import time
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
# 缩放点积注意力
class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, querries, keys, values, valid_lens):
        d = querries.shape[-1]
        scores = (querries @ keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return self.dropout(self.attention_weights) @ values
    
# 掩蔽softmax操作
def masked_softmax(X, valid_lens):
    # X是3D张量 batch-num-dim, valid_lens 为1D或2D张量
    if valid_lens is None:
        return F.softmax(X,dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
            
        X = sequence_mask(X.reshape(-1,shape[-1]), valid_lens, value=-1e6)
        
        return F.softmax(X.reshape(shape), dim=-1)
    
def sequence_mask(X, valid_len, value=0):
    maxlen = X.shape[1]
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

# 转置矩阵QKV使得可以并行计算多头注意力
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])    

# 将计算完的结果再转置回来
def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)



