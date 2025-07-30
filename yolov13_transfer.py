import os

import torch
import torch.nn as nn
from collections import OrderedDict
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import A2C2f,A2C2fv13, AAttn, AAttnv13, ABlock, ABlockv13
from ultralytics import YOLO



# 由于v13并未集成到ultralytics中，相关网络层和ultralytics有冲突，因此我重写了网络层，需要将作者预训练好的权重文件中的网络层替换掉
# 运行本文件要求：同文件夹下存在对应的权重文件 yolov13{l/n/s/x}.pt


def transfer_yolov13(model_file):

    # model = torch.load('yolov13n.pt')
    model = torch.load(model_file)
    # model_new = YOLO('yolov13n.yaml')

    # model_new.load_state_dict(model['model'].state_dict(),strict=False)
    model_old = model['model']
    model_weights = model_old.state_dict()
    model_structure = model_old.model
    model_yaml = model_old.yaml

    # model_yaml修改
    for k,v in model_yaml.items():
        if isinstance(v,list):
            for ll in v:
                if ll[2] in ['A2C2f','ABlock','AAttn']:
                  ll[2] += 'v13'

    # model_sequenstial 修改
    def rename_seq_keys(module, parent=''):
        # 构造新的 OrderedDict
        new_modules = OrderedDict()
        for idx, (k, m, *args) in enumerate(module._modules.items()):
            # print('k={} \t m={}\n'.format(k, m))
            new_m = m
            if m.type == '{}.{}'.format(A2C2f.__module__,A2C2f.__name__):
                mlp = m.m._modules['0']._modules['0'].mlp
                mlp_ratio = mlp._modules['0'].conv.out_channels / mlp._modules['0'].conv.in_channels
                new_m = A2C2fv13(c1=m.cv1._modules['conv'].in_channels, c2=m.cv2._modules['conv'].out_channels, n=len(m.m), a2=True, area=4, residual= not (m.gamma==None), mlp_ratio= mlp_ratio, e=0.5)
                new_m.load_state_dict(m.state_dict(),strict=False)
                new_m.f = m.f
                new_m.i = m.i
                new_m.np = m.np
                new_m.training = m.training
                new_m.type = '{}.{}'.format(A2C2fv13.__module__,A2C2fv13.__name__)
            elif m.type == '{}.{}'.format(AAttn.__module__,AAttn.__name__):
                new_m = AAttnv13(dim=64, num_heads=2, area=4)
                new_m.load_state_dict(m.state_dict(), strict=False)
                new_m.f = m.f
                new_m.i = m.i
                new_m.np = m.np
                new_m.training = m.training
                new_m.type = '{}.{}'.format(AAttnv13.__module__, AAttnv13.__name__)

            elif m.type == '{}.{}'.format(ABlock.__module__,ABlock.__name__):
                new_m = ABlockv13(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
                new_m.load_state_dict(m.state_dict(), strict=False)
                new_m.f = m.f
                new_m.i = m.i
                new_m.np = m.np
                new_m.training = m.training
                new_m.type = '{}.{}'.format(ABlockv13.__module__, ABlockv13.__name__)
            # new_k = '{}v13'.format(k) if k in ['A2C2f','ABlock','AAttn'] else k
            #     m =
            new_modules[k] = new_m

            rename_seq_keys(m)
        module._modules = new_modules

    rename_seq_keys(model_structure)
    model['model'].model = model_structure
    model['model'].yaml = model_yaml
    os.rename(model_file, '{}.backup'.format(model_file))
    torch.save(model, model_file)

if __name__ == '__main__':
    model_file_list = [
        'yolov13l.pt',
        'yolov13n.pt',
        'yolov13s.pt',
        'yolov13x.pt',
                       ]
    for model_file in model_file_list:
        print('model {} begins\n'.format(model_file))
        transfer_yolov13(model_file)
        print('model {} finish\n'.format(model_file))
