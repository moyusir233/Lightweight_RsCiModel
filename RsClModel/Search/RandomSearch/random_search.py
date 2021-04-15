import random
import pandas as pd
from RsClModel.options.base_options import BaseOptions
import os
os.environ['GIT_PYTHON_REFRESH']='quiet'
import torch
from RsClModel.models.wrapper import ModelWrapper
from RsClModel.report import model_summary
import RsClModel.distiller
from RsClModel.thinning import thinning
from RsClModel.models.save_load_models import load
from RsClModel.data import custom_get_dataloaders
def setCompressionScheduler(compression_scheduler, channel_configuration):
    for i, item in enumerate(channel_configuration):
        compression_scheduler.policies[1][i].pruner.desired_sparsity = item
    return compression_scheduler
def randomSearch(net_path,save_path,upper_limit=None,lower_limit=None):
    """随机搜索，需要指定需要保留压缩方案的上下阈值"""
    """==============================环境参数设置==========================="""
    # get options
    opt = BaseOptions().parse()
    if not (upper_limit is None):
        opt.max_rate=upper_limit
    if not (lower_limit is None):
        opt.min_rate = lower_limit
    # basic settings
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids)[1:-1]
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = False
    else:
        device = "cpu"
    ##################### Get Dataloader ####################
    dataloader_train, dataloader_val = custom_get_dataloaders(opt)
    # dummy_input is sample input of dataloaders
    if hasattr(dataloader_val, "dataset"):
        dummy_input = dataloader_val.dataset.__getitem__(0)
        dummy_input = dummy_input[0]
        dummy_input = dummy_input.unsqueeze(0)
    else:
        # for imagenet dali loader
        dummy_input = torch.rand(1, 3, 224, 224)
    #####################  Create Baseline Model  ####################
    net = ModelWrapper(opt)
    # 预训练模型的权重导入
    load(net, net_path)
    flops_before, params_before = model_summary(net.get_compress_part(), dummy_input)
    compression_scheduler = distiller.file_config(
        net.get_compress_part(), net.optimizer, opt.compress_schedule_path
    )
    num_layer = len(compression_scheduler.policies[1])
    #####################  Pruning Strategy Generation ###############
    compression_scheduler = distiller.file_config(
        net.get_compress_part(), net.optimizer, opt.compress_schedule_path
    )
    output=[]
    while len(output)<10:
        # 随机产生压缩向量
        channel_config = []
        for i in range(num_layer):
            channel_config.append(random.uniform(opt.min_rate, opt.max_rate))
        compression_scheduler = setCompressionScheduler(
            compression_scheduler, channel_config
        )
        ###### Adaptive-BN-based Candidate Evaluation of Pruning Strategy ###
        thinning(net, compression_scheduler, input_tensor=dummy_input)
        flops_after, params_after = model_summary(net.get_compress_part(), dummy_input)
        ratio = flops_after / flops_before
        if upper_limit>=ratio>=lower_limit:
            net = net.to(device)
            net.parallel(opt.gpu_ids)
            net.get_compress_part().train()
            with torch.no_grad():
                for index, sample in enumerate(dataloader_train):
                    _ = net.get_loss(sample)
                    if index > 100:
                        break
            strategy_score = net.get_eval_scores(dataloader_val)["accuracy"]
            channel_config.insert(0,strategy_score)
            channel_config.insert(0,ratio)
            output.append(channel_config)
        del net
        net = ModelWrapper(opt)
        # 预训练模型的权重导入
        load(net, net_path)
    columns=["acc","ratio","vars"]
    columns+=[""]*(num_layer-1)
    pd.DataFrame(output,columns=columns).to_csv(save_path)

