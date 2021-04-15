# -*- coding: utf-8 -*-
import os
os.environ['GIT_PYTHON_REFRESH']='quiet'
import torch
import torch.optim as optim
from options.base_options import BaseOptions
from models.wrapper import ModelWrapper
from report import model_summary, Reporter
from data import custom_get_dataloaders
from tqdm import tqdm
import numpy as np
import distiller
from thinning import thinning
from models.save_load_models import save
import pandas as pd
from models.save_load_models import load

def CaluKappa(x):
    total=np.sum(x.value())
    temp1=0;temp2=0;
    for i in range(x.value().shape[0]):
        temp1+=x.value()[i,i]
        temp2+=np.sum(x.value()[i,:])*np.sum(x.value()[:,i])
    p0=temp1/total
    pe=temp2/total/total
    return round((p0-pe)/(1-pe),4)
def setCompressionScheduler(compression_scheduler, channel_configuration):
    for i, item in enumerate(channel_configuration):
        compression_scheduler.policies[1][i].pruner.desired_sparsity = item
    return compression_scheduler

def train_epoch(model_wrapper, dataloader_train, optimizer):
    optimizer.zero_grad()
    model_wrapper._net.train()

    loss_total = 0
    total = 0

    for iter_in_epoch, sample in enumerate(dataloader_train):
        loss = model_wrapper.get_loss(sample)
        loss_total += loss.item()
        total += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if total%100==0:
            print(loss.item())
    return loss_total / total


def main(opt,channel_config,dataloader_train, dataloader_val,path):
    # basic settings
    torch.backends.cudnn.enabled = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids)[1:-1]

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = False
    else:
        device = "cpu"
    ##################### Get Dataloader ####################

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
    load(net,path)
    #net.load_checkpoint(opt.checkpoint)
    #####################  Load Pruning Strategy ###############
    compression_scheduler = distiller.file_config(
        net.get_compress_part(), net.optimizer, opt.compress_schedule_path
    )
    compression_scheduler = setCompressionScheduler(
        compression_scheduler, channel_config
    )
    ###### Adaptive-BN-based Candidate Evaluation of Pruning Strategy ###
    thinning(net, compression_scheduler, input_tensor=dummy_input)
    flops_after, params_after = model_summary(net.get_compress_part(), dummy_input)
    net = net.to(device)
    net.parallel(opt.gpu_ids)
    net.get_compress_part().train()
    t=tqdm(dataloader_train, leave=False)
    with torch.no_grad():
        for index, sample in enumerate(t):
            _ = net.get_loss(sample)
            if index > 100:
                break
    strategy_score = net.get_eval_scores(dataloader_val)["accuracy"]
    old=strategy_score
    print( "Evaluation score:{}".format(strategy_score))
    ##################### Fine-tuning #########################
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(net.optimizer,100,eta_min=5e-5)
    #lr_scheduler=optim.lr_scheduler.StepLR(net.optimizer,5,0.9)
    reporter = Reporter(opt)
    best_acc = strategy_score
    best_kappa = 0
    net._net.train()
    for epoch in range(1,opt.epoch+1):
        net.confusion_matrix.reset()
        reporter.log_metric("lr", net.optimizer.param_groups[0]["lr"], epoch)
        train_loss = train_epoch(net, dataloader_train, net.optimizer,)
        reporter.log_metric("train_loss", train_loss, epoch)
        lr_scheduler.step()
        scores = net.get_eval_scores(dataloader_val)
        kappa = CaluKappa(net.confusion_matrix)
        print("==> Evaluation: Epoch={} Acc={}".format(epoch, str(scores)))
        reporter.log_metric("eval_acc", scores["accuracy"], epoch)
        reporter.log_metric("kappa", kappa, epoch)
        if scores["accuracy"] > best_acc:
            best_acc = scores["accuracy"]
            best_kappa = kappa
            save_checkpoints(
                scores["accuracy"], net._net, reporter, opt.exp_name, epoch,
            )
        reporter.log_metric("best_acc", best_acc, epoch)
        save_checkpoints(
            scores["accuracy"], net._net, reporter, opt.exp_name, epoch,
        )
        print("==> Training epoch %d" % epoch)
    """将模型转换为torch script保存"""
    ckpt_name="{}_best.pth".format(opt.exp_name)
    load(net,os.path.join(reporter.ckpt_log_dir, ckpt_name))
    net._net.eval()
    traced_script_module = torch.jit.trace(net._net, torch.rand(1,3,256,256))
    traced_script_module.save(os.path.join(reporter.log_dir,"model.pt"))
    del net
    return old,best_acc,best_kappa,flops_after,params_after

def save_checkpoints(acc, model, reporter, exp_name, epoch):
    if not hasattr(save_checkpoints, "best_acc"):
        save_checkpoints.best_acc = 0
    state_dict = model.state_dict()
   # reporter.save_checkpoint(state_dict, "{}_latest.pth".format(exp_name), epoch)
    if acc > save_checkpoints.best_acc:
        reporter.save_checkpoint(state_dict, "{}_best.pth".format(exp_name), epoch)
        save_checkpoints.best_acc = acc
        #save(model,"model",reporter.log_dir)
    #reporter.save_checkpoint(state_dict, "{}_{}.pth".format(exp_name, epoch), epoch)
def finetune(opt,input,output):
    # get options
    dataloader_train, dataloader_val = custom_get_dataloaders(opt)
    best_config = pd.read_csv(input)
    channel_config = np.array(best_config)
    print(channel_config.shape)
    if channel_config.shape[1]!=37:
        raise Exception
    result=[]
    pertrain_net_path=opt.net_path
    for i in range(channel_config.shape[0]):
        print(i+1)
        temp=main(opt,channel_config[i,:],dataloader_train, dataloader_val,pertrain_net_path)
        result.append(temp)
    try:
        data = pd.DataFrame(result, columns='before_acc after_acc kappa Flops Params'.split(' '))
        data.to_csv(output)
    except Exception:
        data = pd.DataFrame(result, columns='before_acc after_acc kappa Flops Params'.split(' '))
        data.to_csv('/flag.csv')
if __name__=='__main__':
    opt = BaseOptions().parse()
    inputs=opt.inputs
    outputs=opt.outputs
    for i in range(len(inputs)):
        finetune(opt,inputs[i],outputs[i])


