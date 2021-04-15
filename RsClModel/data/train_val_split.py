# -*- coding: utf-8 -*-
import os
import shutil
import random
def train_val_split(root,flag):#随机取3/10的数据作为验证集
    class_name=os.listdir(root)
    train_path=os.path.join(root,'train')
    val_path=os.path.join(root,'val')
    os.mkdir(train_path)
    os.mkdir(val_path)
    for i in class_name:
        shutil.move(os.path.join(root, i), train_path)
        train_class_path=os.path.join(train_path,i)
        val_class_path=os.path.join(val_path,i)
        os.mkdir(val_class_path)
        random_move(train_class_path,val_class_path,flag)
def random_move(root,target_path,flag):
    file_names=os.listdir(root)
    random.seed(flag)
    file_names=random.sample(file_names, int(len(file_names)*0.3))
    for i in file_names:
        src=os.path.join(root, i)
        shutil.move(src,target_path)
if __name__=='__main__':
    train_val_split('/home/igarss/NWPU-RESISC45',0)