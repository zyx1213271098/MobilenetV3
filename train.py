import argparse
import logging
import torch
import torch.optim as optim
from MobileNetV3 import MobileNetV3_Small
from utils import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def train(opt):
    # ********************** 模型定义和预训练权重加载 **********************
    model = MobileNetV3_Small(num_classes=100)
    print(model.state_dict().keys())
    checkpoint = torch.load(opt.weight)
    new_state_dict = {}
    for key,value in checkpoint['state_dict'].items():
        if key.startswith('module.linear4'):
            continue
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value
    new_state_dict.update(model.state_dict())
    model.load_state_dict(new_state_dict)
    model.cuda()
    print(model)

    # ********************** 优化器定义 **********************
    nominal_batch_size = 64
    accumulate = max((nominal_batch_size / opt.batch_size), 1)  # accumulate befor optimizing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mobilenetv3")

    # ********************** 模型数据 **********************
    parser.add_argument("--weight", type=str, default=r"./pretrain_model/mbv3_small.pth")

    # ********************** 训练过程 **********************
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    # ********************** 优化器 **********************
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--adam", action="store_true", default=True, help="use torch.optim.Adam() optimizer")
    opt = parser.parse_args()
    logger.info("test logger print")
    print("args:", opt)

    # 确定随机种子，reproducible
    init_seeds(seed=0)  # 0, slower, more reproducible

    train(opt)