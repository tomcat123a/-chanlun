# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:34:56 2020

@author: Administrator
"""
import pandas as pd
import numpy as np
import torch
import os
from torch.nn import Conv1d,BatchNorm1d, ModuleList ,Sequential,LSTM,GRU ,Sigmoid
from torch.nn import AdaptiveMaxPool1d,AdaptiveAvgPool1d,Linear,BCELoss ,BCEWithLogitsLoss,MaxPool1d
from torch.nn import ReLU,Dropout,AvgPool1d
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
from resnet1d import BasicBlock,_resnet1d
import argparse
Parser=argparse.ArgumentParser()

Parser.add_argument("--out_channels", type=int,default=10,help='chromosome numner,1-23')
Parser.add_argument("--debug", type=int,default=0,help='1,debug==True,0,debug==False')
Parser.add_argument("--l", type=int,default=1,help='1,days')
Parser.add_argument("--T", type=int,default=1,help='minute intervals 1-30')
parser=Parser.parse_args()
class myDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,  file_path='E:/1m_18/stockdata/spy_1min_20200522.csv' if os.name=='nt'\
            else '../data/spy_1min_20200522.csv',l=1,T=10,\
                standard=True,columns=['close']):
        assert columns
        """
        Args: used
            l days
            columns :['open', 'high', 'low', 'close']
        """
        d=pd.read_csv(file_path,index_col=0)
        d.index=pd.to_datetime( d.index)
        self.d=d
        dayd=d.resample('1D',base=0,label='left',closed='right').agg({'open': 'first', 
                     'high': 'max', 
                     'low': 'min', 
                     'close': 'last'}).dropna()
        self.dayd=dayd
        self.daysused=l
        self.T=T
        self.standard=standard
        self.columns=columns
        dindex=dayd.index
        self.basicl=d.loc[np.logical_and( \
    d.index < dindex[l],d.index>=dindex[0]) ].dropna().shape[0]
    def __len__(self):
        return self.dayd.shape[0]-self.daysused-1

    def __getitem__(self, idx,debug=0):
        dindex=self.dayd.index
        d=self.d
        l=self.daysused
        minute=d.loc[   np.logical_and( \
    d.index<dindex[idx+l] , d.index>=dindex[idx]) ].dropna()
        lastclose=minute['close'].iloc[-1]
        dayd=self.dayd
        T=self.T
        if minute.shape[0]<self.basicl:
             
            basicl=self.basicl
            m1=pd.DataFrame(\
   minute.iloc[-1,-1]*np.ones((basicl-minute.shape[0],minute.shape[1])))
            m1.columns=minute.columns
            minute=pd.concat((minute,m1 ),0)
        y=int( dayd.iloc[idx+l,1]/lastclose>1.007 )
        
        x=minute[self.columns].values[::T].T
        if self.standard:
            zx=(x-np.mean(x))/np.std(x)
        else:
            zx=x
        if np.sum(np.isnan(zx.reshape(1,-1)))>0:
            print(zx)
        return torch.from_numpy(zx.astype(np.float32)) ,torch.tensor(y)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      

    
dataset=myDataset(l=parser.l,T=parser.T,columns=['open', 'high', 'low', 'close']) 
 
 
#dataset.__getitem__(38)

trainloader=DataLoader(dataset,batch_size=int(0.8*dataset.__len__())-1, sampler=\
 SubsetRandomSampler(range(int(0.8*dataset.__len__()))),shuffle=False,drop_last=True)



testloader=DataLoader(dataset,batch_size=dataset.__len__()-int(0.8*dataset.__len__() ), sampler=\
 SubsetRandomSampler(range(int(0.8*dataset.__len__() ),\
 dataset.__len__())),shuffle=False,drop_last=True)





class cnnseq1(torch.nn.Module):  #resnet
    def __init__(self,out_channels,kernel_size,stride,dilation):
        super(cnnseq, self).__init__()
        self.Conv=Conv1d(in_channels=1,out_channels=out_channels,\
        kernel_size=kernel_size,\
        stride=stride,padding=0,dilation=dilation)
        self.pool=AvgPool1d(kernel_size=5,stride=3)
        self.Conv2=Conv1d(in_channels=out_channels,\
    out_channels=out_channels,kernel_size=3,stride=stride,dilation=dilation)
        self.act=ReLU()
        self.pool2=AvgPool1d(kernel_size=3,stride=2)
        self.linear=Linear(out_channels*4,1)
        self.sigm=Sigmoid()
         
    def forward(self, x ):
        
        x=self.act( self.Conv(x) )
        x=self.pool(x)
        x= self.act( self.Conv2(x)  )
        x=self.pool2(x)
         
        x=self.sigm( self.linear(x.flatten(1,-1)).squeeze(-1) )
        return  x

class cnnseq(torch.nn.Module):  #resnet
    def __init__(self,out_channels,kernel_size,stride,dilation):
        super(cnnseq, self).__init__()
        self.l2=Sequential(Linear(130,100),ReLU() , \
    Linear(100,20),ReLU() ,Linear(20,1))
        self.l1= Sequential(Linear(130,40),ReLU()  ,\
    Linear(40,1) )
        self.l0= Sequential(Linear(130,1))
        self.act=Sigmoid()
    def forward(self, x ):
        x=x.squeeze(1)
        x2= self.l2(x).squeeze(-1) 
        x1=self.l1(x).squeeze(-1) 
        x0=self.l0(x).squeeze(-1)
        return  self.act(x2+x1+x0)
    
 
 
''' 
def evaluate(mod,trainloader,device):
     
    mod=mod.eval()
    for tx,ty in trainloader:
        tx=tx.to(device)
        ty=ty.to(device)
         
        out=mod(tx)
        real=trainy.cpu().data.numpy().astype(np.int32).reshape(-1)
        predclass= (out.cpu().data.numpy()>0.5).astype(np.int32)
        tp=len(np.where(np.logical_and(predclass==1,real==1))[0])
        precision=tp/(1e-4+len(np.where(predclass==1)[0]))
        recall=tp/(1e-4+len(np.where(real==1)[0]))
    print('precision {:3f} recall {:3f} perctg {:3f}'.\
        format(precision,recall,np.mean(real))) 

'''
def evaluate1(mod,x,y,device,verbose=0):
     
    mod=mod.eval()
     
         
    out=mod(x)
    real=y.cpu().data.numpy().astype(np.int32).reshape(-1)
    predclass= (out.cpu().data.numpy()>0).astype(np.int32)
    tp=len(np.where(np.logical_and(predclass==1,real==1))[0])
    precision=tp/(1e-4+len(np.where(predclass==1)[0]))
    recall=tp/(1e-4+len(np.where(real==1)[0]))
    if verbose==1:
        print('precision {:.4f} recall {:.4f} perctg {:.3f}'.\
            format(precision,recall,np.mean(real))) 

for tx,ty in trainloader:
    trainx=tx.to(device)
    trainy=ty.to(device).float()
import time

for tx,ty in testloader:
    testx=tx.to(device)
    testy=ty.to(device).float()

mod=_resnet1d('',block=BasicBlock,layers=[2,2])
def train1(mod,trainx,trainy,testx,testy,device,epoch):
    CEL=BCEWithLogitsLoss()
    
    optimizer=torch.optim.Adam(mod.parameters(), lr=0.1,amsgrad=True)
    for it in range(epoch):
        mod=mod.train()
    
        optimizer.zero_grad() 
        out=mod(trainx).squeeze(1) 
        assert out.size()==trainy.size()
        loss=CEL(out,trainy  )
        loss.backward()
        optimizer.step() 
        #print('epoch{:.0f} done loss {:.4f}'.format(it,loss.cpu().data.numpy()))
        mod=mod.eval()
        print('train')
        evaluate1(mod,trainx,trainy,device,0)
        print('test')
        evaluate1(mod,testx,testy,device,0)
t0=time.time() 
train1(mod,trainx,trainy,testx,testy,device,1000)
print(time.time()-t0)
evaluate1(mod,trainx,trainy,device)
evaluate1(mod,testx,testy,device,1)
mod(testx)>0.5
