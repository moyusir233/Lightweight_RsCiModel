import numpy as np
import geatpy as gp
import os
os.environ['GIT_PYTHON_REFRESH']='quiet'
import torch
from models.wrapper import ModelWrapper
from report import model_summary
import distiller
from thinning import thinning
from models.save_load_models import load
def SetCompressionScheduler(compression_scheduler, channel_configuration):
    for i, item in enumerate(channel_configuration):
        compression_scheduler.policies[1][i].pruner.desired_sparsity = item
    return compression_scheduler
def get_eval(opt,net_path,args):
    #####################  Create Baseline Model  ####################
    net = ModelWrapper(opt)
    load(net,net_path)
    #####################  Pruning Strategy Generation ###############
    compression_scheduler = distiller.file_config(
        net.get_compress_part(), net.optimizer, opt.compress_schedule_path
    )
    compression_scheduler = SetCompressionScheduler(
        compression_scheduler, args["channel_config"]
    )
    ###### Adaptive-BN-based Candidate Evaluation of Pruning Strategy ###
    thinning(net, compression_scheduler, input_tensor=args["dummy_input"])
    flops_after, params_after = model_summary(net.get_compress_part(), args["dummy_input"])
    ratio = flops_after / args["flops_before"]
    net = net.to(args["device"])
    net.parallel(opt.gpu_ids)
    net.get_compress_part().train()
    with torch.no_grad():
        for index, sample in enumerate(args["dataloader_train"]):
            _ = net.get_loss(sample)
            if index > 100:
                break
    strategy_score = net.get_eval_scores(args["dataloader_val"])["accuracy"]
    del net
    return 1-strategy_score,ratio
class MoeaProblem(gp.Problem):
    def __init__(self,opt,pertrain_model_path,args):
        self.opt=opt
        self.pertrain_model_path=pertrain_model_path
        self.args=args
        name='MyProblem'
        M=2#目标维数
        maxormins=[1,1]#1:min,-1:max
        Dim=args["num_layer"]#决策变量维数
        varTypes=[0]*Dim#决策变量类型,0连续,1离散
        lb=[opt.min_rate]*Dim#决策变量下界
        ub=[opt.max_rate]*Dim#决策变量上界
        lbin=[1]*Dim#决策变量下边界,0表示不包含下界，1表示包含
        ubin=[1]*Dim#决策变量上边界
        #调用父类构造方法完成实例化
        super().__init__(name,M,maxormins,Dim,varTypes,lb,ub,lbin,ubin)
    def aimFunc(self,pop):#pop为传入的种群对象
        Vars=pop.Phen#得到种群的决策变量矩阵，每一行代表一个个体，每一列代表一个决策变量
        pop_size=Vars.shape[0]#种群的个体数量
        #cv=np.ones((pop_size,1))#可行性约束矩阵，每一行代表一个个体，每一列代表一个约束条件
        objv=np.ones((pop_size,2))#目标函数矩阵，每一行代表一个个体，每一列由各个个体的目标函数值组成
        for i in range(pop_size):
            channel_config=Vars[i,:]
            # 该函数返回不进行微调的剪枝网络在验证集上的acc与该剪枝网络的剪枝率，前者为目标函数值，后者为约束
            score,ratio=get_eval(self.opt,self.pertrain_model_path,self.args)
            objv[i,0]=score
            objv[i,1]=ratio
            # 约束:opt.flops_target-0.005<=ratio<=opt.flops_target+0.005
            #cv[i,0]=np.abs(self.opt.flops_target-ratio)-0.005
        pop.ObjV=objv
        #pop.CV=cv
class SoeaProblem(gp.Problem):
    def __init__(self,opt,pertrain_model_path,args):
        self.opt=opt
        self.pertrain_model_path=pertrain_model_path
        self.args=args
        name='MyProblem'
        M=1#目标维数
        maxormins=[1]#1:min,-1:max
        Dim=args["num_layer"]#决策变量维数
        varTypes=[0]*Dim#决策变量类型,0连续,1离散
        lb=[opt.min_rate]*Dim#决策变量下界
        ub=[opt.max_rate]*Dim#决策变量上界
        lbin=[1]*Dim#决策变量下边界,0表示不包含下界，1表示包含
        ubin=[1]*Dim#决策变量上边界
        #调用父类构造方法完成实例化
        super().__init__(name,M,maxormins,Dim,varTypes,lb,ub,lbin,ubin)
    def aimFunc(self,pop):#pop为传入的种群对象
        Vars=pop.Phen#得到种群的决策变量矩阵，每一行代表一个个体，每一列代表一个决策变量
        pop_size=Vars.shape[0]#种群的个体数量
        #cv=np.ones((pop_size,1))#可行性约束矩阵，每一行代表一个个体，每一列代表一个约束条件
        objv=np.ones((pop_size,1))#目标函数矩阵，每一行代表一个个体，每一列由各个个体的目标函数值组成
        for i in range(pop_size):
            channel_config=Vars[i,:]
            # 该函数返回不进行微调的剪枝网络在验证集上的acc与该剪枝网络的剪枝率，前者为目标函数值，后者为约束
            score,ratio=get_eval(self.opt,self.pertrain_model_path,self.args)
            objv[i,0]=score
            #objv[i,1]=ratio
            # 约束:opt.flops_target-0.005<=ratio<=opt.flops_target+0.005
            #cv[i,0]=np.abs(self.opt.flops_target-ratio)-0.005
        pop.ObjV=objv
        #pop.CV=cv

