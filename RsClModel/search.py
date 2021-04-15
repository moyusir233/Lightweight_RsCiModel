from Search import *
import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="搜索模型剪枝方案")
    parser.add_argument("--method",choices=["Moea","Soea","Random"],default="Moea",help="选择搜索模型压缩方案的方式")
    parser.add_argument("--net_path",required=True,help="预训练模型权重的保存路径")
    parser.add_argument("--save_path",required=True,help="搜索到的压缩方案的保存目录")
    parser.add_argument("--upper_limit",type=float,help="对每一个卷积层滤波器所允许剪枝的最大比例")
    parser.add_argument("--lower_limit",type=float,help="对每一个卷积层滤波器所允许剪枝的最小比例")
    parser.parse_args()
    method=parser.method
    net_path=parser.net_path
    save_path=parser.save_path
    upper_limit=parser.upper_limit
    lower_limit=parser.lower_limit
    if method=="Moea":
        searchMoea(net_path, save_path,upper_limit,lower_limit)
    elif method=="Soea":
        searchSoea(net_path,save_path,upper_limit,lower_limit)
    else:
        randomSearch(net_path,save_path,upper_limit,lower_limit)