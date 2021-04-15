#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 13 JULY 2020
#  ------------------------------------------------------------------
import os
os.environ['GIT_PYTHON_REFRESH']='quiet'
import torch
import torch.optim as optim
from options.base_options import BaseOptions
from models.wrapper import ModelWrapper
from report import Reporter
from data import custom_get_dataloaders
from models.save_load_models import load


def random_compression_scheduler(compression_scheduler, channel_configuration):
    for i, item in enumerate(channel_configuration):
        compression_scheduler.policies[1][i].pruner.desired_sparsity = item
    return compression_scheduler


def get_channel_config(path, line_num):
    # line_num starts from 0
    with open(path) as data:
        lines = data.readlines()
        i = 0
        for l in lines:
            if i == line_num:
                d = l.strip().split(" ")
                channel_config = []
                print("=" * 20, " read config")
                for i in range(0, 2):
                    print("{} ".format(d[i]), end="")
                for i in range(2, len(d)):
                    channel_config.append(float(d[i]))
                break
            i += 1
    return channel_config


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
    return loss_total / total


def main(opt,dataloader_train, dataloader_val,path=None):
    # basic settings
    torch.backends.cudnn.enabled = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids)[1:-1]

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = False
    else:
        device = "cpu"
    #####################  Create Baseline Model  ####################
    net = ModelWrapper(opt)
    if not path is None:load(net,path)
    #net.load_checkpoint()
    #net=torch.load('/root/Desktop/res50_flop73_0.752.pth')
    net = net.to(device)
    net.parallel(opt.gpu_ids)
    net.get_compress_part().train()
    ##################### Fine-tuning #########################
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(net.optimizer,50,eta_min=5e-6)
    #lr_scheduler=optim.lr_scheduler.StepLR(net.optimizer,10,0.8)
    reporter = Reporter(opt)
    #best_acc = net.get_eval_scores(dataloader_val)["accuracy"]
    best_acc=0
    net._net.train()
    for epoch in range(1, opt.epoch+1):
        reporter.log_metric("lr", net.optimizer.param_groups[0]["lr"], epoch)
        train_loss = train_epoch(net, dataloader_train, net.optimizer)
        reporter.log_metric("train_loss", train_loss, epoch)
        lr_scheduler.step()
        scores = net.get_eval_scores(dataloader_val)
        print("==> Evaluation: Epoch={} Acc={}".format(epoch, str(scores)))
        reporter.log_metric("eval_acc", scores["accuracy"], epoch)
        if scores["accuracy"] > best_acc:
            best_acc = scores["accuracy"]
        reporter.log_metric("best_acc", best_acc, epoch)
        save_checkpoints(
            scores["accuracy"], net._net, reporter, opt.exp_name, epoch,
        )
        print("==> Training epoch %d" % epoch)
def save_checkpoints(acc, model, reporter, exp_name, epoch):
    if not hasattr(save_checkpoints, "best_acc"):
        save_checkpoints.best_acc = 0
    state_dict = model.state_dict()
    reporter.save_checkpoint(state_dict, "{}_latest.pth".format(exp_name), epoch)
    if acc > save_checkpoints.best_acc:
        reporter.save_checkpoint(state_dict, "{}_best.pth".format(exp_name), epoch)
        save_checkpoints.best_acc = acc
    reporter.save_checkpoint(state_dict, "{}_{}.pth".format(exp_name, epoch), epoch)
def train():
    # get options
    opt = BaseOptions().parse()
    print('lr:',opt.lr,'weight_decay:',opt.weight_decay)
    dataloader_train, dataloader_val = custom_get_dataloaders(opt)
    main(opt,dataloader_train,dataloader_val,opt.net_path)
if __name__=='__main__':
    train()
