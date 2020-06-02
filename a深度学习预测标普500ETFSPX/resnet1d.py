# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:31:32 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:18:26 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
 
import torch
import time
from torch.nn import  ModuleList ,BatchNorm1d,Sequential,\
 Linear,MSELoss ,AvgPool1d,Conv1d,AdaptiveAvgPool1d
import xgboost as xgb
#import torch.nn.SyncBatchNorm as BatchNorm1d 
from torch.nn import   ReLU,Dropout
#from hyperopt import hp,tpe,Trials,fmin,STATUS_OK#pip install hyperopt --user
import torch.nn as nn

'''
test

x=torch.ones(5,4,12)
m1=_resnet1d('d',BasicBlock,[1,1],channel_list=[6,12]) 
m2=_resnet1d('d',Bottleneck,[1,1],channel_list=[6,12])
assert  m1(x).size()==torch.Size([5, 1])


m3=CnnModule(in_channel=4,out_channel=10,cnn_ker=3,cnn_stride=1,\
            pool_stride=2,pool_ker=3 ,bn=1)
m3(x).size()

m4=ReduceModule(4)
y=torch.ones(5,30,4,12)
m4(x).size()

m5=CnnDnn()
m5(y).size()
'''

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x1 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, activation='relu'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        if activation=='relu':
            self.relu = nn.ReLU(inplace=True)
        if activation=='gelu':
            self.relu = GELU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, activation='relu'):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x1(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        if activation=='relu':
            self.relu = nn.ReLU(inplace=True)
        if activation=='gelu':
            self.relu = GELU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1d(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,channel_list=[64,128],stride_list=[2,2],\
                 conv_ker=3,\
                 conv_stride=2,pool_ker=3,\
                firstpool_stride=2,activation='relu'):
        super(ResNet1d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.activation=activation
        self.inplanes = channel_list[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False]*len(channel_list)
        if len(replace_stride_with_dilation) != len(channel_list):
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a {}-element tuple, got {}".format(len(channel_list),replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(4, self.inplanes, kernel_size=conv_ker, \
                               stride=conv_stride, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=pool_ker,\
        stride=firstpool_stride, padding=1)
        self.layer_list=ModuleList()
        for i in range(len(channel_list)):
            self.layer_list.append(self._make_layer(block, channel_list[i], \
                            layers[i],dilate=None))
         
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channel_list[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks,stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,activation=self.activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
        base_width=self.base_width, dilation=self.dilation,
       norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        #x.size() =batch,channel,length
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layer_list)):
            x=self.layer_list[i](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet1d(arch, block, layers,**kwargs):
    model = ResNet1d(block, layers, **kwargs)
    
    return model

class CnnModule(torch.nn.Module):
    def __init__(self, in_channel,out_channel,cnn_ker,cnn_stride,\
            pool_stride,pool_ker ,bn):
        super(CnnModule, self).__init__()
        mylist=ModuleList()
        mylist.append(Conv1d(in_channel,out_channel,
                             kernel_size=cnn_ker, \
                               stride=cnn_stride, padding=0,
                               bias=False))
        if bn==1:
            mylist.append(BatchNorm1d(out_channel))
        if pool_ker>0:
            mylist.append(
            AvgPool1d(kernel_size=pool_ker,stride=pool_stride))
        self.mylist=Sequential(*mylist)
    
    def forward(self,x):
        return self.mylist(x)


class CnnDnn(torch.nn.Module):
    def __init__(self, in_channel=4,out_channel=4,cnn_ker=3,cnn_stride=1,\
            pool_stride=2,pool_ker=3,custom=False,bn=1,dnn=[30,1]):
        super(CnnDnn, self).__init__()
        mylist=ModuleList()
        
        if isinstance(cnn_ker,list)==False:
            mylist.append(CnnModule(in_channel=in_channel,\
        out_channel=out_channel,cnn_ker=cnn_ker,cnn_stride=cnn_stride,\
            pool_stride=pool_stride,pool_ker=pool_ker,bn=1))
        else:
            
            for i in range(len(cnn_ker)):
                mylist.append(CnnModule(\
  in_channel=in_channel[i],\
        out_channel=out_channel[i],cnn_ker=cnn_ker[i],\
  cnn_stride=cnn_stride[i],\
            pool_stride=pool_stride[i],pool_ker=pool_ker[i],bn=bn))
        if ~custom:
            mylist.append(\
            ReduceModule(out_channel if \
    ~isinstance(out_channel,list) else out_channel[-1],mean=~custom))
        self.mylist=Sequential(*mylist)
        self.fullyconnected=Fc(encoder_hidden_size=dnn, bn=bn,dr_p=0)
    
    def forward(self,x):
        y=torch.stack( [ self.mylist(x[:,i,:,:]) for i in range(x.size()[1]) ] ,dim=1)
        return self.fullyconnected(y)

class ReduceModule(torch.nn.Module):
    def __init__(self, in_channel,mean=False):
        super(ReduceModule, self).__init__()
        if mean==False:
            self.linear=Linear(in_channel,1)
        else:
            self.linear=AdaptiveAvgPool1d(1)
        self.pool=AdaptiveAvgPool1d(1)
         
    def forward(self,x):
        #x=n,c,l
        x=self.pool(x) 
        x=x.transpose(-1,-2)
        x=self.linear(x) 
        return x.squeeze(-1).squeeze(-1)#n



class LinearBlock(torch.nn.Module):
    def __init__(self, in_channel,out_channel,bn,dr_p ,no_tail=False):
        super(LinearBlock, self).__init__()
         
         
        mylist=ModuleList()
        mylist.append(Linear( in_channel,out_channel))
        if no_tail==False:
            if bn==1:
                mylist.append(BatchNorm1d(out_channel) )
            
            mylist.append(ReLU())
            if dr_p>0:
                mylist.append(Dropout(dr_p) )
         
        self.block= Sequential(*mylist) 
         
         
    def forward(self, x):
        
        return  self.block(x)



class Fc(torch.nn.Module):
    def __init__(self, encoder_hidden_size, bn,dr_p):
        super(Fc, self).__init__()
        self.encoder=ModuleList()
        
        self.n_encoders=len(encoder_hidden_size)-1 
        for i in range(self.n_encoders):
            self.encoder.append( LinearBlock(encoder_hidden_size[i],\
                                             encoder_hidden_size[i+1], bn,dr_p,\
    no_tail=False if encoder_hidden_size[i+1]!=1 else True) )
             
        self.full_model=Sequential(*self.encoder)
    def forward(self, x):
        
        return  self.full_model(x).squeeze(1)#batch
