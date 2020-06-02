# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:27:37 2020

@author: Administrator
"""
import os
os.chdir('D:\\pysrc') 
from rnnst import npyDataset
import torch
from torch.utils.data import DataLoader 
from rnnst import lin,modcnn,train
from torch.utils.data.sampler import SubsetRandomSampler
from resnet1d import _resnet1d 
import argparse
Parser=argparse.ArgumentParser()
Parser.add_argument("--seqlen", type=int,default=480,help='length of bars')
Parser.add_argument("--batch_size", type=int,default=8,help='batch_size')
 
parser=Parser.parse_args()
#dataset=npyDataset('../data/000001.XSHG',Tlist=['1T'],l=parser.seqlen)
dataset=npyDataset('E:/1m_18/000001.XSHG',Tlist=['1T'],\
l=parser.seqlen)
dataset.__len__()
dataset.__getitem__(0)

trainloader=DataLoader(dataset,batch_size=parser.batch_size, sampler=\
SubsetRandomSampler(range(int(0.8*dataset.__len__()))),drop_last=True)
testloader=DataLoader(dataset,batch_size=parser.batch_size, sampler=\
SubsetRandomSampler(range(int(0.8*dataset.__len__() ),\
dataset.__len__()-2)),drop_last=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
 
mod=lin(channels=[1,40,80],dnn=[len(dataset.Tlist)*(parser.seqlen-1)])
train(mod,trainloader,testloader,device,100)
