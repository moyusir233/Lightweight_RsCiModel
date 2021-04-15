# -*- coding: utf-8 -*-
import geatpy as gp
from problem import SoeaProblem,MoeaProblem
import os
os.environ['GIT_PYTHON_REFRESH']='quiet'
import torch
from RsClModel.options.base_options import BaseOptions
from RsClModel.models.wrapper import ModelWrapper
from RsClModel.report import model_summary
from RsClModel.data import custom_get_dataloaders
import RsClModel.distiller
from RsClModel.models.save_load_models import load
def searchMoea(net_path,save_path,upper_limit=None,lower_limit=None):
    """多目标优化搜索,输入为预训练模型的权重所在路径，输出为遗传算法搜索结果的.csv保存的目录路径"""
    """==============================环境参数设置和问题类的初始化==========================="""
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
    #####################  定义问题，并进行遗传算法 ####################
    args = dict()
    args["dataloader_train"] = dataloader_train
    args["dataloader_val"] = dataloader_val
    args["dummy_input"] = dummy_input
    args["flops_before"] = flops_before
    args["device"] = device
    args["num_layer"] = num_layer
    problem = MoeaProblem(opt, net_path,args)
    """==============================种群设置==========================="""
    Encoding='RI'#编码方式,此处为整数实数混合编码
    NIND=10 #种群规模
    Field=gp.crtfld(Encoding, problem.varTypes, problem.ranges,problem.borders)#创建区域描述器
    population = gp.Population(Encoding, Field, NIND)
    """===========================算法参数设置=========================="""
    #myAlgorithm = gp.moea_awGA_templet(problem, population)
    #myAlgorithm = gp.moea_MOEAD_DE_templet(problem, population)
    #myAlgorithm = gp.moea_RVEA_RES_templet(problem, population)
    myAlgorithm = gp.moea_NSGA2_archive_templet(problem, population)
    #实例化算法模板
    myAlgorithm.MAXGEN = 100 # 最大进化代数
    #myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
    #myAlgorithm.recOper.XOVR = 0.7  # 设置交叉概率
    myAlgorithm.logTras = 10  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画)
    """==========================根据先验知识创建先知种群==============="""
    #data = pd.read_csv('/eagle_eye/search_results/old_dataset/pop_get_search_result_moea2/Phen.csv')
    #well_config=np.array(data)[5:,:]
    #prophetpop=gp.Population(Encoding,Field,5,well_config)# 实例化种群对象
    #myAlgorithm.call_aimFunc(prophetpop)# 计算先知种群的目标函数值及约束（假如有约束）
    #myAlgorithm.call_aimFunc()# 计算先知种群的目标函数值及约束（假如有约束）
    """==========================调用算法模板进行种群进化==============="""
    #[BestIndi, population] = myAlgorithm.run(prophetpop)  # 执行带先验种群的算法模板，得到最优个体以及最后一代种群
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    BestIndi.save(os.path.join(save_path,"BestIndi"))  # 把最优个体的信息保存到文件中
    population.save(os.path.join(save_path,"Population"))# save population
    """=================================输出结果======================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过%s秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最优的目标函数值为： ')
        print(BestIndi.ObjV[0,:])
        print('最优的控制变量值为：')
        print(BestIndi.Phen[0,:])
        return BestIndi.ObjV[0,:],BestIndi.Phen[0,:]
    else:
        print('没找到可行解。')
        return -1,-1
#单目标优化搜索
def searchSoea(net_path,save_path,upper_limit=None,lower_limit=None):
    """单目标优化搜索,输入为预训练模型的权重所在路径，输出为遗传算法搜索结果的.csv保存的目录路径"""
    """==============================环境参数设置和问题类的初始化==========================="""
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
    #####################  定义问题，并进行遗传算法 ####################
    args = dict()
    args["dataloader_train"] = dataloader_train
    args["dataloader_val"] = dataloader_val
    args["dummy_input"] = dummy_input
    args["flops_before"] = flops_before
    args["device"] = device
    args["num_layer"] = num_layer
    problem = SoeaProblem(opt, net_path,args)
    """==============================种群设置==========================="""
    Encoding='RI'#编码方式,此处为整数实数混合编码
    NIND=10 #种群规模
    Field=gp.crtfld(Encoding, problem.varTypes, problem.ranges,problem.borders)#创建区域描述器
    population = gp.Population(Encoding, Field, NIND)
    """===========================算法参数设置=========================="""
    myAlgorithm = gp.soea_DE_best_1_L_templet(problem, population)
    #实例化算法模板
    myAlgorithm.MAXGEN = 100 # 最大进化代数
    myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
    myAlgorithm.recOper.XOVR = 0.7  # 设置交叉概率
    myAlgorithm.logTras = 10  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画)
    """==========================根据先验知识创建先知种群==============="""
    #data = pd.read_csv('/eagle_eye/search_results/old_dataset/pop_get_search_result_moea2/Phen.csv')
    #well_config=np.array(data)[5:,:]
    #prophetpop=gp.Population(Encoding,Field,5,well_config)# 实例化种群对象
    #myAlgorithm.call_aimFunc(prophetpop)# 计算先知种群的目标函数值及约束（假如有约束）
    #myAlgorithm.call_aimFunc()# 计算先知种群的目标函数值及约束（假如有约束）
    """==========================调用算法模板进行种群进化==============="""
    #[BestIndi, population] = myAlgorithm.run(prophetpop)  # 执行带先验种群的算法模板，得到最优个体以及最后一代种群
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    BestIndi.save(os.path.join(save_path,"BestIndi"))  # 把最优个体的信息保存到文件中
    population.save(os.path.join(save_path,"Population"))# save population
    """=================================输出结果======================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过%s秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最优的目标函数值为： ')
        print(BestIndi.ObjV[0,:])
        print('最优的控制变量值为：')
        print(BestIndi.Phen[0,:])
        return BestIndi.ObjV[0,:],BestIndi.Phen[0,:]
    else:
        print('没找到可行解。')
        return -1,-1
