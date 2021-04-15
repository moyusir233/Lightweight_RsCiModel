# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:42:05 2020

@author: 27997
"""
import torch
import os
def save(module,module_name,path):
    name=module_name +'.pth'
    torch.save(module,os.path.join(path,name))
def load(module,path,flag=1):
    if flag:
        temp=dict()
        for k,v in torch.load(path)['state_dict'].items():
            if not k.startswith('_net.'):
                new_key='_net.'+k
                temp[new_key]=v
        module.load_state_dict(temp)
    else:
        temp=torch.load(path)
        module.load_state_dict(temp)
