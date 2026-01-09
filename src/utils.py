import torch

def get_std_opt(model, d_model):
    # 返回纯粹的 Adam 优化器，学习率计算交给 train.py
    return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)