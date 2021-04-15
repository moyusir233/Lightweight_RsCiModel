#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 13 JULY 2020
#  ------------------------------------------------------------------
import argparse
import os


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # model params
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="resnet50",
            help="what kind of model you are using. Only support `resnet50`, `mobilenetv1` and `mobilenetv1_imagenet`",
        )
        self.parser.add_argument(
            "--num_classes", type=int, default=45, help="num of class label"
        )
        # env params
        self.parser.add_argument(
            "--gpu_ids", type=int, default=[0], nargs="+", help="GPU ids."
        )

        # fine-tune params
        self.parser.add_argument(
            "--batch_size", type=int, default=128, help="batch size while fine-tuning"
        )
        self.parser.add_argument(
            "--epoch", type=int, default=50, help="epoch while fine-tuning"
        )
        self.parser.add_argument(
            "--dataset_path", type=str, default="/home/igarss/NWPU-RESISC45", help="path to dataset"
        )
        self.parser.add_argument(
            "--dataset_name",
            type=str,
            default="imagenet",
            help="filename of the file contains your own `get_dataloaders` function",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers used in dataloading",
        )
        self.parser.add_argument(
            "--lr", type=float, default=1e-4, help="learning rate while fine-tuning"
        )
        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=8e-4,
            help="weight decay while fine-tuning",
        )
        self.parser.add_argument(
            "--momentum", type=float, default=0.9, help="momentum while fine-tuning"
        )
        self.parser.add_argument("--log_dir", type=str, default="logs/", help="log dir")
        self.parser.add_argument(
            "--exp_name", type=str, default="exp", help="experiment name"
        )

        # search params
        self.parser.add_argument(
            "--max_rate", type=float, default=0.8, help="define search space"
        )
        self.parser.add_argument(
            "--min_rate", type=float, default=0.01, help="define search space"
        )
        self.parser.add_argument(
            "--compress_schedule_path",
            type=str,
            default="compress_config/res50_imagenet.yaml",
            help="path to compression schedule",
        )
        self.parser.add_argument("--net_path", help="预训练模型权重的保存路径")
        self.parser.add_argument("--inputs", action="append",help="压缩方案的保存路径")
        self.parser.add_argument("--outputs", action="append",help="压缩结果的保存路径")
        self.initialized = True
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
