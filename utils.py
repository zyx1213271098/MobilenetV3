import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True  # 确定性算法
        cudnn.benchmark = False  # 不选最快实现方法
    else:   # faster, less reproducible
        cudnn.deterministic = False  # 非确定性算法
        cudnn.benchmark = True  # 不选择最快实现方法
