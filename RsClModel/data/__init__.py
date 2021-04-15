#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 13 JULY 2020
#  ------------------------------------------------------------------
import os
import torch
import importlib
from .imagenet import get_dataloaders
def custom_get_dataloaders(opt):
    return get_dataloaders(opt.batch_size, opt.num_workers, path=opt.dataset_path)
