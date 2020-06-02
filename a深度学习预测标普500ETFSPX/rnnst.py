# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:48:56 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
import torch
from torch.nn import Conv1d,BatchNorm1d, ModuleList ,Sequential,LSTM,GRU ,Sigmoid
from torch.nn import AdaptiveMaxPool1d,AdaptiveAvgPool1d,Linear,BCELoss ,MSELoss,MaxPool1d
from torch.nn import ReLU
from torch.utils.data import Dataset
from datetime import timedelta
 


class period(torch.nn.Module):  #resnet
    def __init__(self,input_size,hidden_size,num_layers):
        super(period, self).__init__()
        self.rnn=LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=1 ,batch_first=True,bidirectional=True)

    def forward(self, x ):
        y,_=self.rnn(x)
        return y[:,-1,:] #batch,2*hiddent_size
    
class modrnn(torch.nn.Module):  #resnet
    def __init__(self,input_size,hidden_size,num_layers,l):
        super(modrnn, self).__init__()
        self.rnnlist=ModuleList()
        for i in range(l):
            self.rnnlist.append(\
   period(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers))
        self.l=l    
        self.fc=Sequential(BatchNorm1d(hidden_size*2*l),Linear(hidden_size*2*l,l),ReLU(),\
            BatchNorm1d(l),Linear(l,1),Sigmoid())
    def forward(self, x ):#x.size()=batch,len,channel(4)
        if self.l==1:
            y=self.rnnlist[0](x)
        else:
            y=self.rnnlist[0](x[:,0,:,:])
        for j in range(1,self.l):
            y=torch.cat((y,self.rnnlist[j](x[:,j,:,:])),dim=-1 )
        z=self.fc(y)
        
        return z#y.size()=batch,hidden_size*2*self.l
    
class modcnn(torch.nn.Module):  #resnet
    def __init__(self,channels,dnn ):
        super(modcnn, self).__init__()
        self.rnnlist=ModuleList()
        
        for i in range(len(channels)-1):
            if i < len(channels)-1-1:
                self.rnnlist.append(Sequential(Conv1d(channels[i],channels[i+1],kernel_size=5),\
        ReLU(),BatchNorm1d(channels[i+1]),MaxPool1d(kernel_size=5,stride=3)))
            else:
                self.rnnlist.append(Sequential(Conv1d(channels[i],channels[i+1],kernel_size=5),\
    ReLU(),BatchNorm1d(channels[i+1]),AdaptiveAvgPool1d(1)))
        self.cnn=Sequential(*self.rnnlist)
        if len(dnn)==1:
            self.fc=Sequential(BatchNorm1d(channels[-1]),Linear(channels[-1],1),Sigmoid())
        else:
            self.fc=ModuleList()
            self.fc.append(Sequential(BatchNorm1d(channels[-1]),Linear(channels[-1],dnn[0]),ReLU()))
            for j in range(len(dnn)-1):
                self.fc.append(Sequential(BatchNorm1d(dnn[j]),\
  Linear(dnn[j],dnn[j+1]),(ReLU() if j < len(dnn)-2 else Sigmoid())))
            self.fc=Sequential(*self.fc)
    def forward(self, x ):#x.size()=batch,len,channel(4)
        x=x.unsqueeze(1)#x.size()=batch,c,l
       # print(x.size())
        y=self.cnn(x).squeeze(-1)
        #print(y.size())
         
        z=self.fc(y).squeeze(-1)
        
        return z#y.size()=batch,hidden_size*2*self.l




class lin(torch.nn.Module):  #resnet
    def __init__(self,channels,dnn ):
        super(lin, self).__init__()
        self.linear=Sequential(Linear(dnn[0],1),Sigmoid())
    def forward(self, x ):#x.size()=batch,len,channel(4)
        x=self.linear(x)#x.size()=batch,c,l
        return x.squeeze(-1)
class npyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,  file_path,Tlist=['1T'],l=200):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            note:
            redf must contain a key '1D',but,only keys in Tlist will be 
            returned by __getitiem
        """
        self.file_path=file_path
        self.df=pd.read_csv(file_path,index_col=0)
        
        self.df.index=pd.to_datetime(self.df.index)
        self.redf={}
        self.l=l
        self.Tlist=Tlist
        for t in Tlist:
            self.redf[t]=\
        self.df.resample(t,base=0,label=('right' if t[-1]!='D' else 'left'),closed='right').agg({'open': 'first', 
             'high': 'max', 
             'low': 'min', 
             'close': 'last'}).dropna()
        if not ('1D' in Tlist):
            t='1D'
            self.redf[t]=\
        self.df.resample(t,base=0,label=('right' if t[-1]!='D' else 'left'),closed='right').agg({'open': 'first', 
             'high': 'max', 
             'low': 'min', 
             'close': 'last'}).dropna()
        self.days=self.redf['1D'].shape[0]-3
    def __len__(self):
        return self.days

    def __getitem__(self, idx,debug=0,minute=30):
        todayidx=self.l+idx#idx for self.redf['1D']
        
        todaydate=self.redf['1D'].iloc[todayidx].name
        tomorrowdate=self.redf['1D'].iloc[todayidx+1].name
        if debug==1:
            print(todaydate)
            print(tomorrowdate)
        now=todaydate+timedelta(days=0,hours=14,minutes=minute)
        out={}#dict of kbars for different periods
        tomorrow_start_time=tomorrowdate+timedelta(days=0,hours=9,minutes=31)
        tomorrow_end_time=tomorrowdate+timedelta(days=0,hours=14,minutes=minute)
        try:
            for t in self.Tlist:
                if t[-1]!='D':
                    out[t]=self.redf[t].loc[:now].tail(self.l).values
                else:
                    out[t]=self.redf[t].loc[:now].iloc[-self.l-1:-1].values
        except Exception as e:
            print(e)
            print(idx)
        close=self.redf['1T'].loc[now]['close']
        lowrate=self.redf['1T'].loc[now:tomorrow_end_time]['low'].min()/close-1
        highrate=self.redf['1T'].loc[tomorrow_start_time:tomorrow_end_time]['high'].max()/close-1 
        for t in self.Tlist:
            out[t]=   out[t][1:,:] /out[t][:-1,-1].reshape(-1,1)-1  
        return torch.from_numpy(100*np.concatenate([out[k][:,-1] for k in out.keys()],\
        axis=-1).astype(np.float32)),\
        torch.tensor(np.float32(highrate))


def train(mod,trainloader,testloader,device,epoch):
    CEL=MSELoss()
    optimizer=torch.optim.Adam(mod.parameters(), lr=0.01,amsgrad=True)
    for it in range(epoch):
        mod=mod.train()
        for tx,ty in trainloader:
             
            tx=tx.to(device)
            ty=ty.to(device)
            optimizer.zero_grad() 
            out=mod(tx)
            
            loss=CEL(out,ty)
            loss.backward()
            optimizer.step() 
        print('epoch{} done'.format(it))
        mod=mod.eval()
        print('train')
        evaluate(mod,trainloader,device)
        print('test')
        evaluate(mod,testloader,device)




def evaluate(mod,trainloader,device):
     
    mod=mod.eval()
    for tx,ty in trainloader:
        tx=tx.to(device)
        ty=ty.to(device)
         
        out=mod(tx)
        real=ty.cpu().data.numpy().astype(np.int).reshape(-1)
        pred=out.cpu().data.numpy().astype(np.int).reshape(-1)
        R2=1-np.mean((real-pred)**2)/np.mean( (real-np.mean(real))**2)
        meanerr=np.mean(abs(real-pred))
        maxerr=np.max(abs(real-pred))
        meantrue=np.mean( real) 
        meanpred=np.mean( pred) 
    print('R^2: {},mean(abs(err))={},max(abs(err))={},meantrue={},meanpred={}'.\
        format(R2,meanerr,\
        maxerr,meantrue,meanpred))  

