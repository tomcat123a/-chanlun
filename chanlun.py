# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:32:43 2020

@author: Administrator
"""

import sys
import pandas as pd
import numpy as np
from  datetime import date
import os
 

  
data_dir='/scratch/tmp/pudge/chan/data/'
            


#end debug
#init from non-inclusion
def buy_sell(INDEX,data_dir,debug=1):
    os.chdir(data_dir)
    len_dir = os.listdir(data_dir)
 
    if date.fromtimestamp(os.path.getmtime(len_dir[INDEX]))<date.today():#if the file is
         #not updated on today
         return None
# =============================================================================
    if debug == 0:
        debug = 1
    df = pd.read_csv(len_dir[INDEX])[['low','high','datetime']][:-debug]
    if debug >= len(df):
        print('skipped')
        return ;
    print('processing ' + len_dir[INDEX].split('_')[1].split('.')[0])
    i = 0
    while(True ):
        if ( df['low'][i] <=  df['low'][i+1] ) or (df['high'][i] <=  df['high'][i+1]):
            i = i + 1
        else :
            break
    df = df[i:].reset_index(drop=True)
    
    #REMOVE INCLUSION
    while ( True ):
        temp_len = len(df)
        i=0
        while i<=len(df)-4:
            if (df.iloc[i+2,0]>=df.iloc[i+1,0] and df.iloc[i+2,1]<=df.iloc[i+1,1]) or\
            (df.iloc[i+2,0]<=df.iloc[i+1,0] and df.iloc[i+2,1]>=df.iloc[i+1,1]):
                if df.iloc[i+1,0]>df.iloc[i,0]:
                    df.iloc[i+2,0] = max(df.iloc[i+1:i+3,0])
                    df.iloc[i+2,1] = max(df.iloc[i+1:i+3,1])
                    df.drop(df.index[i+1],inplace=True)
                    
                    continue
                else:
                    df.iloc[i+2,0] = min(df.iloc[i+1:i+3,0])
                    df.iloc[i+2,1] = min(df.iloc[i+1:i+3,1])
                    df.drop(df.index[i+1],inplace=True)
                    
                    continue
            i = i + 1
       # print(len(df))    
        if len(df)==temp_len:
            break
            
    df= df.reset_index(drop=True)  
    #get difenxing and dingfenxing
    ul=[0]
    for i in range(len(df)-2):
        if df.iloc[i+2,0] < df.iloc[i+1,0] and df.iloc[i,0] < df.iloc[i+1,0]:
            ul = ul + [1]
            continue
        if df.iloc[i+2,0] > df.iloc[i+1,0] and df.iloc[i,0] > df.iloc[i+1,0]:
            ul = ul + [-1]# difenxing -1 dingfenxing +1
            continue
        else:
            ul = ul + [0]
    ul = ul + [0]
    global df1
    df1 = pd.concat((df[['low','high']],pd.DataFrame(ul),df['datetime']),axis=1)
      
    i = 0
    
    while df1.iloc[i,2] == 0 and i < len(df1)-2:
        i = i + 1
    df1=df1[i:]  
    
    i = 0
    while ( sum(abs(df1.iloc[i+1:i+4,2]))>0 or df1.iloc[i,2]==0) and i < len(df1)-2:
        i = i + 1
    df1=df1[i:]
    df1.rename(columns= {0:'od'},inplace=True)
    #df1.columns=Index(['low', 'high', 'od', 'datetime'], dtype='object')
    #df1.columns=Index(['low', 'high', 'od', 'datetime'], dtype='object')
    #df1.columns=Index(['low', 'high', 'od', 'datetime'], dtype='object')
    #df1.columns=Index(['low', 'high', 'od', 'datetime'], dtype='object')
    if len(df1)<=60:
        print('error!')
        return ;
    #remove those within 3 bars
    df1=df1.reset_index(drop=True)
    global od_list#od_list are the index of df1 whose corresponding point are fenxing extreme vertex
    
    od_list=[0]
    judge(0,0,1) 
    
    
    #judge(27,34,-1)
    
    
    
    
    
    #generate seg
    start = 0
    while start < len(od_list)-5:
        if check_init_seg(od_list[start:start+4]):
            break
        else:
            start = start + 1
    
    lines = []
    
    i = start
    end = False
    while i <= len(od_list)-4:
        se = Seg(od_list[i:i+4])
        label = False
        while label == False and i <= len(od_list)-6:
            i = i + 2
            label,start = se.grow(od_list[i+2:i+4])
            if se.vertex[-1] > od_list[-3]:
                end =True
                
                lines += [se.lines()]
                break
        if end:
            break
        i =  np.where(np.array(od_list) == se.vertex[-1])[0][0]
        #show datetime of the end of the segment
        #print(df1.iloc[se.vertex[-1],3])   
        lines += [se.lines()]#there are still remaining fewer than or equal to 
        #3 bies not considered in the last
        #seg ,which is unfinished and named by tails
    low_list=df1.iloc[se.vertex[-1]:,0]
    high_list=df1.iloc[se.vertex[-1]:,1]
    
    low_extre=low_list.min()
    high_extre=high_list.max()
    if se.finished == True:
        if lines[-1][0][1] < lines[-1][1][1] :#d==1
            lines += [ [(se.vertex[-1],lines[-1][1][1]),(low_list.idxmin(),low_extre)]]
        else:
            lines += [ [(se.vertex[-1],lines[-1][1][1]),(high_list.idxmax(),high_extre)]]
            
    else:
        if lines[-1][0][1] < lines[-1][1][1] :#d==1
            if low_extre > lines[-1][0][1]:
                lines[-1] = [ (lines[-1][0][0],lines[-1][0][1]),(high_list.idxmax(),high_extre)] 
            else:
                if low_list.idxmin()-se.vertex[-1]>=10:
                    lines += [ [(se.vertex[-1],lines[-1][1][1]),(low_list.idxmin(),low_extre)]]
        else:
            
            if high_extre < lines[-1][0][1]:
                lines[-1] = [ (lines[-1][0][0],lines[-1][0][1]),(low_list.idxmin(),low_extre) ]
            else:
                if  high_list.idxmax()-se.vertex[-1]>=10:
                    lines += [ [(se.vertex[-1],lines[-1][1][1]),(high_list.idxmax(),high_extre)]]   
    
    #print(lines)
    #tails is the unfinished seg,tails[4] is its direction
    a,tails = get_pivot(lines)    
    pro_a= process_pivot(a)
# =============================================================================
#     if len(pro_a)>=4:
#         if pro_a[-1].trend==-1 and pro_a[-2].trend==0 and pro_a[-3].trend==-1 and\
#         tails[4]==-1 and pro_a[-1].finished ==0 and df1.iloc[-1][0] <pro_a[-1].dd :
#             for yi in range(0,len(a)):
#                 pro_a[yi].dis1()
# =============================================================================
    
    
    signal,interval = buy_point1(pro_a,tails)    
    if signal:#trend slow down, first pivot dd > next pivot gg
        pro_a[-1].write_out('../buy1/'+len_dir[INDEX].split('_')[1].split('.')[0]+'_buy1.txt',tails)
        
    signal,interval = buy_point3_des(pro_a,tails)    
    if signal:#trend slow down, first pivot dd > next pivot gg
        pro_a[-1].write_out('../buy3/'+len_dir[INDEX].split('_')[1].split('.')[0]+'_buy3.txt',tails)
    signal,interval = buy_point23(pro_a,tails)    
    if signal:#trend slow down, first pivot dd > next pivot gg
         pro_a[-1].write_out('../buy23/'+len_dir[INDEX].split('_')[1].split('.')[0]+'_buy23.txt',tails)
    signal,interval = buy_point2(pro_a,tails)    
    if signal:#trend slow down, first pivot dd > next pivot gg
         pro_a[-1].write_out('../buy2/'+len_dir[INDEX].split('_')[1].split('.')[0]+'_buy2.txt',tails)
    signal,interval = sell_point1(pro_a,tails)    
    if signal:#trend slow down, first pivot dd > next pivot gg
        pro_a[-1].write_out('../sell1/'+len_dir[INDEX].split('_')[1].split('.')[0]+'_sell1.txt',tails)
        
    signal,interval = sell_point3_ris(pro_a,tails)    
    if signal:#trend slow down, first pivot dd > next pivot gg
        pro_a[-1].write_out('../sell3/'+len_dir[INDEX].split('_')[1].split('.')[0]+'_sell3.txt',tails)
    signal,interval = sell_point2(pro_a,tails)    
    if signal:#trend slow down, first pivot dd > next pivot gg
         pro_a[-1].write_out('../sell2/'+len_dir[INDEX].split('_')[1].split('.')[0]+'_sell2.txt',tails)
     

#end buy_sell


#utility
def same_d(a1,a2,b1,b2,a_sign):
    #a1 low a2 high b1 low b2 high
    if a_sign == 1:
        return (a1 > b1 and a2 > b2)
    else:
        return (a1 < b1 and a2 < b2)
def new_extreme(a1,a2,b1,b2,a_sign):
    #whether b has new extreme than a,true return true
    if a_sign == 1:
        return b2 >= a2
    else:
        return a1 >= b1
    
def write_seg(temp_lines,file,buy_sign,interval):
    if buy_sign==True:
        f = open(file,'w')
        f.write('seg-3:'+str(df1.iloc[temp_lines[-3][0][0],3])+' '+\
                str(df1.iloc[temp_lines[-3][0][0],1])+str(df1.iloc[temp_lines[-3][1][0],3])+' '+\
                str(df1.iloc[temp_lines[-3][1][0],0])  )
        f.write('\n')
        f.write('seg-2:'+str(df1.iloc[temp_lines[-2][0][0],3])+' '+
                str(df1.iloc[temp_lines[-2][0][0],0])+str(df1.iloc[temp_lines[-2][1][0],3])+' '+
                str(df1.iloc[temp_lines[-2][1][0],1]) )
        f.write('\n')
        f.write('seg-1:'+str(df1.iloc[temp_lines[-1][0][0],3])+' '+
                str(df1.iloc[temp_lines[-1][0][0],1])+str(df1.iloc[temp_lines[-1][1][0],3])+' '+
                str(df1.iloc[temp_lines[-1][1][0],0]) )
        f.write('cur_price:\n'+str(df1.iloc[-1,0]))
        f.write('\n')
        f.write('cur_time:\n'+str(df1.iloc[-1,3]))
        f.write('\n')
        if df1.iloc[temp_lines[-1][1][0],0]<interval:
            f.write('target_price:'+str(interval))
        else:
            f.write('supp_price:'+str(interval))
        
        f.close()
    else:
        f = open(file,'w')
        f.write('seg-3:'+str(df1.iloc[temp_lines[-3][0][0],3])+' '+\
                str(df1.iloc[temp_lines[-3][0][0],0])+str(df1.iloc[temp_lines[-3][1][0],3])+' '+\
                str(df1.iloc[temp_lines[-3][1][0],1])  )
        f.write('\n')
        f.write('seg-2:'+str(df1.iloc[temp_lines[-2][0][0],3])+' '+
                str(df1.iloc[temp_lines[-2][0][0],1])+str(df1.iloc[temp_lines[-2][1][0],3])+' '+
                str(df1.iloc[temp_lines[-2][1][0],0]) )
        f.write('\n')
        f.write('seg-1:'+str(df1.iloc[temp_lines[-1][0][0],3])+' '+
                str(df1.iloc[temp_lines[-1][0][0],0])+str(df1.iloc[temp_lines[-1][1][0],3])+' '+
                str(df1.iloc[temp_lines[-1][1][0],1]) )
        f.write('cur_price:\n'+str(df1.iloc[-1,1]))
        f.write('\n')
        f.write('cur_time:\n'+str(df1.iloc[-1,3]))
        f.write('\n')
        if df1.iloc[temp_lines[-1][1][0],0]>interval:
            f.write('target_price:'+str(interval))
        else:
            f.write('resist_price:'+str(interval))
        
        f.close()
def exist_opposite(cur_i,d,pos):
    #print("exist_opposite")
    #print('e0'+str(cur_i+pos))
        #print('e1'+str(df1.iloc[cur_i+pos,0]))
    return df1['od'].iloc[cur_i+pos]==-d and same_d(df1.iloc[cur_i,0],df1.iloc[cur_i,1],\
     df1.iloc[cur_i+pos,0],df1.iloc[cur_i+pos,1],d)

def exist_new_extreme(cur_i,d,start,end):
    j = start
    while j <= end:
        if new_extreme(df1.iloc[cur_i,0],df1.iloc[cur_i,1],df1.iloc[cur_i + j,0],df1.iloc[cur_i + j,1],d):
            return cur_i + j,True
        j = j + 1
    return cur_i,False



def judge(prev_i,cur_i,d):#d the direction of fenxing to be confirmed, prev_i the previous confirmed
    #d == df1['od][cur_i] should hold when finished and prev_i = cur_i is set
    global od_list 
    
    
    
    #print('start ' + str(cur_i))
    if cur_i + 4 >= len(df1)-1:
        #print('finished')
        
        #stop()
        return 0
        
    if cur_i - prev_i < 4 or df1['od'].iloc[cur_i] != d:
        cur_i = cur_i + 1
        #print(cur_i)
        judge(prev_i,cur_i,d)
    else:# at least 4 bars later and direction correct
        # now df1['od'].iloc[cur_i] ==d and cur_i - prev_i >= 4 
        
        new_i,label1 = exist_new_extreme(cur_i,d,2,3)
        if label1 == True:
            cur_i = new_i
            #print("f1")
            judge(prev_i,cur_i,d)
        else:
            k = 4
            if cur_i  + k + 1>= len(df1)-1:
                #print ("finishe2!")
                return 0
            
            while not exist_opposite(cur_i,d,k):
            #while True:    
                #kth >=4 later bar does not match opposite fenxing
                new_i,label2 = exist_new_extreme(cur_i,d,k,k)
                if label2 == True:
                    cur_i = new_i
                    judge(prev_i,cur_i,d)
                    return 0
                    #print('f2')
                else:
                    k = k + 1
                    if cur_i  + k >= len(df1)-1:
                        #print ("finishe4!")
                        return 0
                
            #confirmed by existent opposite fenxing
            prev_i = cur_i
            cur_i = cur_i + k
            od_list = od_list + [prev_i]
            #print('added' + str(prev_i))
            #print('input ' + str(cur_i))
            #print('-d ' + str(d))
            judge(prev_i,cur_i,-d)
    #print('post call judge' + str(cur_i))
#end judge



#utils for seg
    

def check_init_seg(start_l):
    #return True successful False fail
    d = -df1.iloc[start_l[0],2]
    if not ((d == 1 or d == -1 )and(len(start_l)==4)):
        print('initializing seg failed in  check_init_seg!')
            
    if d == 1:
        if df1.iloc[start_l[1],1] < df1.iloc[start_l[3],1] and \
        df1.iloc[start_l[0],0] < df1.iloc[start_l[2],0]:#valid
            return True
        else:
            return False
    else:
        if df1.iloc[start_l[1],0] > df1.iloc[start_l[3],0] and \
        df1.iloc[start_l[0],1] > df1.iloc[start_l[2],1]:#valid
            return True
        else:
            return False

class Seg:

    # Initializer / Instance Attributes
    #directino of a seg is the same as its first bi,
    #direction of a bi is the negative of its starting fenxing:
    #rising bi is +1 and falling bi is -1
    def __init__(self, start_l):
        
        self.start = start_l[0]
        
        if df1.iloc[start_l[0],2]==0:
            print("error init!")
        self.d = - df1.iloc[start_l[0],2]
        
        self.finished = False
        self.vertex = start_l
        self.gap = False
        if self.d == 1:
            self.cur_extreme =  df1.iloc[start_l[3],1]
            self.cur_extreme_pos =  start_l[3]
            self.prev_extreme =  df1.iloc[start_l[1],1]
        else:
            self.cur_extreme =  df1.iloc[start_l[3],0]
            self.cur_extreme_pos =  start_l[3]
            self.prev_extreme =  df1.iloc[start_l[1],0]
    
    def grow(self,new_l):
        #len(new_l) == 2
        #two consecutive bis will be added
        #new_d, direction of the first bi added
        if 1 == self.d:#rising seg
            if df1.iloc[new_l[1],1] >= self.cur_extreme:#new extreme
                if df1.iloc[new_l[0],0] > self.prev_extreme:
                    self.gap = True
                else:
                    self.gap = False
                self.prev_extreme = self.cur_extreme
                self.cur_extreme = df1.iloc[new_l[1],1]
                self.cur_extreme_pos =  new_l[1]
                
            else:# no new extreme two cases to finish
                if (self.gap == False and df1.iloc[new_l[1],0] < df1.iloc[self.vertex[-1],0]) or \
                (self.gap == True and (df1.iloc[self.vertex[-1],1] < df1.iloc[self.vertex[-3],1] ) \
                 and (df1.iloc[self.vertex[-2],0] < df1.iloc[self.vertex[-4],0] )):
                    self.finished = True
                    
                    self.vertex = [ i for i in self.vertex if i <= self.cur_extreme_pos]
                    #print("finished")
                    #print(self.vertex)
                    #print(self.getrange())
                    return True,self.vertex[-1]
                
                    
            #seg continued
            self.vertex = self.vertex + new_l
            
            return False,0
            
        else:
            if df1.iloc[new_l[1],0] <= self.cur_extreme:#new extreme
                if df1.iloc[new_l[0],1] < self.prev_extreme:
                    self.gap = True
                else:
                    self.gap = False
                self.vertex = self.vertex + new_l
                self.prev_extreme = self.cur_extreme
                self.cur_extreme = df1.iloc[new_l[1],0]
                self.cur_extreme_pos =  new_l[1]
            else:# no new extreme two cases to finish
                if (self.gap == False and df1.iloc[new_l[1],1] > df1.iloc[self.vertex[-1],1]) or \
                (self.gap == True and (df1.iloc[self.vertex[-1],0] > df1.iloc[self.vertex[-3],0] ) \
                 and (df1.iloc[self.vertex[-2],1] > df1.iloc[self.vertex[-4],1] )):
                    self.finished = True
                    
                    self.vertex = [ i for i in self.vertex if i <= self.cur_extreme_pos]
                    #print("finished")
                    #print(self.vertex)
                    #print(self.getrange())
                    return True,self.vertex[-1]
            
            #seg continued    
            self.vertex = self.vertex + new_l
            
            return False,0
    def check_finish(self):
        #two consecutive bis will be added
        #new_d, direction of the first bi added
        if len(self.vertex)-1 <= 5:
            print("no need to check!")
            return False
        
    def getrange(self):
        if self.d == 1:
            return [ df1.iloc[self.start,0],self.cur_extreme,self.d ]
        else:
            return [ df1.iloc[self.start,1],self.cur_extreme,self.d ]                
    def show(self):
        print(self.vertex)
        print( self.getrange() )
        print(df1.iloc[self.vertex[-1],3])
    def lines(self):#lines :d==1 ==> [(index in df1 of starting
        #,low of line),(index of end ,high of line)],
        #d==-1 ==>[(,high of line),(,low of line)]
        return [(self.start,self.getrange()[0]),\
                (self.vertex[-1],self.getrange()[1])]
    
#end class Seg
#each object of pivot is a pivot
#1min pivot
class Pivot1:
    def __init__(self, lines,d):#lines a 3 element list of Seg.getlines()
        
        self.trend = -2
        self.level = 1
        self.enter_d = d#
        self.aft_l_price = 0
        self.aft_l_time = '00'# time for third type buy or sell point
        self.future_zd = -float('inf')
        self.future_zg = float('inf')
        if d == 1:#pivot=[zg,zd,dd,gg,start_time,end_time,d] d the direction of
                #the seg pre-entering but not in pivot
            if lines[3][1][1] <= lines[1][0][1]: #low of line i+3 < low of line i+1   
                self.zg = min(lines[1][0][1],lines[3][0][1])
                self.zd = max(lines[3][1][1],lines[1][1][1])
                self.dd = lines[2][0][1]
                self.gg = max(lines[1][0][1],lines[2][1][1])
                      
        else:#pivot=[zg,zd,dd,gg,start_time,end_time,start_seg_index,end_seg_index,d] d the seg pre-entering pivot
            if lines[3][1][1] >= lines[1][0][1]:    
                self.zg = min(lines[1][1][1],lines[3][1][1])
                self.zd = max(lines[3][0][1],lines[1][0][1])
                self.dd = min(lines[2][1][1],lines[1][0][1])
                self.gg = lines[2][0][1]
        
        self.start_index = lines[1][0][0]
        self.end_index = lines[2][1][0]# should be updated after growing 
        #lines[self.end_index] is the leaving seg
        self.finished = 0
        self.enter_force = seg_force(lines[0])
        self.leave_force = seg_force(lines[3])# should be updated after growing
        self.size  = 3#should be updated
        self.mean = 0.5*(self.zd + self.zg)
        self.start_time = df1.iloc[self.start_index,3 ]
        self.leave_start_time = df1.iloc[self.end_index,3 ]# should be updated after growing
        self.leave_end_time = df1.iloc[lines[3][1][0],3 ] # should be updated after growing
        self.leave_d = -d # should be updated after growing
        self.leave_end_price = lines[3][1][1] # should be updated after growing
        self.leave_start_price = lines[3][0][1]
        self.prev2_force = seg_force(lines[1])
        self.prev1_force = seg_force(lines[2])
        self.prev2_end_price = lines[1][1][1]
        #tail_price the leave seg's end price,if the seg
        #is still not finished,its leave seg is the last seg within the pivot
        
    def grow(self,seg):#seg a Seg.getlines()
        
        self.prev2_force = self.prev1_force 
        self.prev1_force = self.leave_force
        self.prev2_end_price = self.leave_start_price
        if seg[1][1] > seg[0][1]:#d for the line is 1
            if (seg[1][1]>=self.zd and seg[0][1] <= self.zg) and (self.size <=28):#then the seg is
                # added to the pivot
                self.end_index = seg[0][0]
                
                self.size = self.size + 1
                self.dd = min(self.dd,seg[0][1])
                
                self.leave_force = seg_force(seg)
                self.leave_start_time = df1.iloc[self.end_index,3 ]
                self.leave_end_time = df1.iloc[seg[1][0],3 ]
                self.leave_d = 2*int(seg[1][1]>seg[0][1])-1
                self.leave_start_price = seg[0][1]
                self.leave_end_price = seg[1][1]
                
                if self.size in [4,7,10,19,28]:#level expansion
                    self.future_zd = max(self.future_zd ,self.dd)
                    self.future_zg = min(self.future_zg ,self.gg)
                
                if self.size in [10,28]:#level expansion
                    self.level = self.level + 1
                    self.zd = self.future_zd
                    self.zg = self.future_zg
                    self.future_zd = -float('inf')
                    self.future_zg = float('inf')
                
            else:
                
                if (seg[1][1]>=self.zd and seg[0][1] <= self.zg):
                    self.dd = min(self.dd,seg[0][1])
                    self.finished = 0.5        
                else:
                    self.finished = 1
                
                
                self.aft_l_price = seg[1][1]
                self.aft_l_time = df1.iloc[seg[1][0],3]
                #only when the seg is finished is the tail_price different from end_price
        else:#d for the line is -1. falling line
            if (seg[1][1]<=self.zg and seg[0][1] >= self.zd) and self.size<=28:#then the seg is
                # added to the pivot
                self.end_index = seg[0][0]
                self.end_price = seg[0][1]
                self.size = self.size + 1
                self.gg = max(self.gg,seg[0][1])
                
                self.leave_force = seg_force(seg)
                self.leave_start_time = df1.iloc[self.end_index,3 ]
                self.leave_end_time = df1.iloc[seg[1][0],3 ]
                self.leave_d = 2*int(seg[1][1]>seg[0][1])-1
                self.leave_start_price = seg[0][1]
                self.leave_end_price = seg[1][1]
                
                if self.size in [4,7,10,19,28]:#level expansion
                    self.future_zd = max(self.future_zd ,self.dd)
                    self.future_zg = min(self.future_zg ,self.gg)
                
                if self.size in [10,28]:#level expansion
                    self.level = self.level + 1
                    self.zd = self.future_zd
                    self.zg = self.future_zg
                    self.future_zd = -float('inf')
                    self.future_zg = float('inf')
            else:
                
                
                if (seg[1][1]<=self.zg and seg[0][1] >= self.zd) :#broke because it is too long
                    self.gg = max(self.gg,seg[0][1])
                    self.finished = 0.5   
                else:
                    self.finished = 1
                
                
                self.aft_l_price = seg[1][1]
                self.aft_l_time = df1.iloc[seg[1][0],3]
        
    
    def display(self):
        print('enter_d:'+str(self.enter_d))
        print('zd:'+str(self.zd))
        print('zg:'+str(self.zg))
        print('dd:'+str(self.dd))
        print('gg:'+str(self.gg))
        print('start_index:'+str(self.start_index))
        print('end_index:'+str(self.end_index))
        print('start_time:'+str(self.start_time))
        
        print('size:'+str(self.size))
        print('enter_force:'+str(self.enter_force))
        print('leave_force:'+str(self.leave_force))
        print('finished:'+str(self.finished))
        print('leave_start_time:'+str(self.leave_start_time))
        print('leave_end_time:'+str(self.leave_end_time))
        print('leave_d:'+str(self.leave_d))
        print('leave_start_price:'+str(self.leave_start_price))
        print('leave_end_price:'+str(self.leave_end_price))
        print('mean:'+str(self.mean))
        print('aft_l_price:'+str(self.aft_l_price))
        
    def dis1(self):
        print('trend:'+str(self.trend))
        print('level:'+str(self.level))
        print('enter_d:'+str(self.enter_d))
        print('zd:'+str(self.zd))
        print('zg:'+str(self.zg))
        print('dd:'+str(self.dd))
        print('gg:'+str(self.gg))
        print('leave_d:'+str(self.leave_d))
        print('start_time:'+str(self.start_time))
        print('leave_start_time:'+str(self.leave_start_time))
        print('\n')
    
    def write_out(self,filepath,extra=''):
        f = open(filepath,'w')
        f.write(' zd:' + str(self.zd)+' zg:'+str(self.zg) +
                ' dd:' + str(self.dd)+' gg:'+str(self.gg) +
                ' leave_d:' + str(self.leave_d)+
                ' prev2_leave_force:' +str(self.prev2_force)+ ' leave_force:' + str(self.leave_force)+
                '\n  start_time:'+str(self.start_time)+
                '  leave_start_time:'+str(self.leave_start_time)+
                '  leave_end_time:'+str(self.leave_end_time)+
                '  prev2_end_price:'+str(self.prev2_end_price)+
                '  leave_end_price:'+str(self.leave_end_price)+
                '\n  size: ' + str(self.size)+' finished: ' + str(self.finished) + ' trend:' + 
                str(self.trend) + ' level:' + 
                str(self.level))
        f.write('\n')
        if extra!='':
            f.write('tails:')
            f.write(str(extra))
            f.write('\n')
            f.write('now')
            f.write(str(df1.iloc[-1]))
        f.close()    
        return 
        
                
#ebd class Pivot
def seg_force(seg):
    return 1000*abs(seg[1][1]/seg[0][1]-1)/(seg[1][0]-seg[0][0])
    
    
def get_pivot(lines):
    Pivot1_array = []
    i = 0
    
    while i < len(lines):
        #print(i)
        d =  2 * int( lines[i][0][1] < lines[i][1][1] ) - 1
        if i < len(lines)-3:
            if d == 1:#pivot=[zg,zd,dd,gg,start_time,end_time,d] d the direction of
                #the seg pre-entering but not in pivot
                if lines[i+3][1][1] <= lines[i+1][0][1]: #low of line i+3 < low of line i+1   
                    pivot = Pivot1(lines[i:i+4],d)
                    i_j = 1
                    while i + i_j < len(lines)-3 and pivot.finished == 0:
                        pivot.grow(lines[i + i_j + 3])
                        i_j = i_j +1
                    
                    
                    i = i + pivot.size 
                    Pivot1_array = Pivot1_array + [pivot]
                    continue
                else:
                    i = i + 1
                    
            else:#pivot=[zg,zd,dd,gg,start_time,end_time,start_seg_index,end_seg_index,d] d the seg pre-entering pivot
                if lines[i+3][1][1] >= lines[i+1][0][1]:    
                    pivot = Pivot1(lines[i:i+4],d)
                    i_j = 1
                    while i + i_j < len(lines)-3 and pivot.finished == 0:
                        pivot.grow(lines[i + i_j + 3])
                        i_j = i_j +1
                    i = i + pivot.size 
                    
                    Pivot1_array = Pivot1_array + [pivot]
                    continue
                else:
                    i = i + 1
                    
        else:
            i = i + 1
            
        
        #pivot [zd,zg,dd,gg] zd and zg may not be valid after expansion            
        # the second para returned is the tails,or the last unconfirmed seg,with tails[4] its d        
    return Pivot1_array , [df1.iloc[lines[-1][0][0],3],lines[-1][0][1],\
                   df1.iloc[lines[-1][1][0],3],lines[-1][1][1],2*int(lines[-1][1][1]>lines[-1][0][1])-1]



           
#same hierachy decomposition
#def process_pivot(pivot)  :
#    i = 0
#    while i < len(pivot)-1:
#        if min(pivot[i][2:4]) <= max(pivot[i+1][2:4])  and\
#        max(pivot[i][2:4]) >= min(pivot[i+1][2:4]):
#            pivot[i+1][2] = min(pivot[i][2],pivot[i+1][2])
#            pivot[i+1][3] = max(pivot[i][3],pivot[i+1][3])
#            pivot[i+1][4] =pivot[i][4]
#            pivot[i+1][5] = pivot[i+1][5]
#            del pivot[i]
#        else:
#            i = i + 1
#    return pivot

def process_pivot(pivot):
    for i in range(0,len(pivot)-1):
        if pivot[i ].level==1 and pivot[i+1].level==1:
            if pivot[i].dd > pivot[i+1].gg:
                pivot[i+1].trend=-1
            else:
    	         if pivot[i].gg < pivot[i+1].dd:
    	             pivot[i+1].trend=1
    	         else:
    	             pivot[i+1].trend=0
        else:
            if pivot[i ].gg> pivot[i +1].gg and pivot[i ].dd> pivot[i +1].dd:
                pivot[i+1].trend=-1
            else:
                if pivot[i ].gg < pivot[i +1].gg and pivot[i ].dd < pivot[i +1].dd:
                    pivot[i+1].trend=1
                else:
                    pivot[i+1].trend=0
    return pivot





def buy_point1(pro_pivot,tails,num_pivot=2):
    if len(pro_pivot)<=3 or tails[4]==1 or pro_pivot[-1].size>=8 or pro_pivot[-1].finished!=0 \
    or df1.iloc[-1][0]/pro_pivot[-1].leave_end_price -1>0 or \
     df1.iloc[-1][0] > tails[3]:
        return False,0
    else:#two pivot descending
        #no slow down
        if ( pro_pivot[-1].prev2_end_price >pro_pivot[-1].leave_end_price ) and \
        (pro_pivot[-1].leave_start_time==tails[0]) and\
             df1.iloc[-1][0] < pro_pivot[-1].dd and \
            1.2*pro_pivot[-1].leave_force <pro_pivot[-1].prev2_force and \
            ( pro_pivot[-1].dd >pro_pivot[-1].leave_end_price ):
            return True,pro_pivot[-1].dd#target price
        else:
            return False,0  
# =============================================================================
#         if pro_pivot[-1].finished == 1 or pro_pivot[-1].leave_d!=-1 or \
#         pro_pivot[-1].finished == 0.5:
#             return False,0
#         if num_pivot == 2:
#             if pro_pivot[-2].enter_d==-1 and pro_pivot[-1].gg < pro_pivot[-2].dd \
#             and pro_pivot[-1].enter_d==-1 and  tails[0]==pro_pivot[-1].leave_start_time \
#             and tails[3] <= pro_pivot[-1].dd:#tails[0] tail seg start_time
#                 return True,pro_pivot[-1].zd
#             else:
#                 return False,0  
#         if num_pivot == 3:
#             if pro_pivot[-3].enter_d==-1 and pro_pivot[-2].gg < pro_pivot[-3].dd and \
#             pro_pivot[-2].enter_d==-1 and pro_pivot[-1].gg < pro_pivot[-2].dd and\
#             pro_pivot[-1].enter_d==-1 and  \
#              pro_pivot[-1].leave_force<\
#             pro_pivot[-1].enter_force and  tails[0]==pro_pivot[-1].leave_start_time \
#             and tails[3] <= pro_pivot[-1].dd:
#                 return True,pro_pivot[-1].zd
#             else:
#                 return False,0    
# =============================================================================

def buy_point2(pro_pivot,tails,num_pivot=2):
    if len(pro_pivot)<=3 or tails[4]==1 or pro_pivot[-1].size>=8 or pro_pivot[-1].finished!=0 \
    or df1.iloc[-1][0]/pro_pivot[-1].leave_end_price -1>0 or \
     df1.iloc[-1][0] > tails[3]:
        return False,0
    else:#two pivot descending
        #no slow down
        if ( pro_pivot[-1].prev2_end_price <pro_pivot[-1].leave_end_price ) and \
        (pro_pivot[-1].leave_start_time==tails[0]) and\
             pro_pivot[-1].prev2_end_price == pro_pivot[-1].dd and \
             pro_pivot[-1].leave_start_price >0.51*(pro_pivot[-1].zd+pro_pivot[-1].zg) :
            return True,pro_pivot[-1].prev2_end_price#support price
        else:
            return False,0


          
def buy_point3_des(pro_pivot,tails):
    if len(pro_pivot)<=2 or(tails[4]==1) or (pro_pivot[-1].finished!=1) or \
    pro_pivot[-1].level > 1 or df1.iloc[-1][0]/pro_pivot[-1].leave_end_price -1>0 or \
     df1.iloc[-1][0] > tails[3]:
        return False,0
    else:#two pivot descending
        if df1.iloc[-1][0] <0.98*pro_pivot[-1].leave_end_price  and df1.iloc[-1][0] >1.02*pro_pivot[-1].zg and \
        pro_pivot[-1].aft_l_price >1.02*pro_pivot[-1].zg and \
         tails[0] == pro_pivot[-1].leave_end_time and \
         pro_pivot[-1].leave_force > pro_pivot[-1].prev2_force and\
         pro_pivot[-1].leave_end_price > pro_pivot[-1].prev2_end_price:
            return True,pro_pivot[-1].zg#support price
        else:
            return False,0
             #no slow down
# =============================================================================
#         if not np.mean( pro_pivot[-3][0:2]) / np.mean(pro_pivot[-2][0:2]) > \
#         np.mean(pro_pivot[-2][0:2]) / np.mean(pro_pivot[-1][0:2]):
#             return False,0
#         if num_pivot == 3:
#             if pro_pivot[-4][2]>pro_pivot[-3][2] and  pro_pivot[-3][2]>pro_pivot[-2][2] \
#             :
#                 return True,pro_pivot[-1][0]
#             else:
#                 return False,0  
#         if num_pivot == 2:
#             if pro_pivot[-3][2]>pro_pivot[-2][2] and pro_pivot[-2][2]>pro_pivot[-1][2] \
#             :
#                 return True,pro_pivot[-1][0]
#             else:
#                 return False,0  
# =============================================================================
def buy_point23(pro_pivot,tails):
    if len(pro_pivot)<=3 or pro_pivot[-1].finished!=1 or \
    pro_pivot[-1].level > 1 or df1.iloc[-1][0]/pro_pivot[-1].leave_end_price -1>0 or \
     df1.iloc[-1][0] > tails[3]:
        return False,0
    else:#two pivot descending
        #no slow down
        if df1.iloc[-1][0] <0.98*pro_pivot[-1].leave_end_price and df1.iloc[-1][0] >1.01*pro_pivot[-1].zg and pro_pivot[-1].trend==-1 \
        and tails[3] >1.01*\
         pro_pivot[-1].zg and tails[0] == pro_pivot[-1].leave_end_time  and \
         pro_pivot[-1].leave_start_price ==pro_pivot[-1].dd:
             return True,pro_pivot[-1].zg# support price
        else:
            return False,0 
       

def sell_point1(pro_pivot,tails,num_pivot=2):
    if len(pro_pivot)<=3 or tails[4]==-1 or pro_pivot[-1].size>=8 or pro_pivot[-1].finished!=0\
     or df1.iloc[-1][1]/pro_pivot[-1].leave_end_price -1<0 or \
     df1.iloc[-1][0] < tails[3]:
        return False,0
    else:#two pivot descending
        #no slow down
        if ( pro_pivot[-1].prev2_end_price <pro_pivot[-1].leave_end_price ) and \
        (pro_pivot[-1].leave_start_time==tails[0]) and\
             df1.iloc[-1][0] > pro_pivot[-1].zg and \
            1.2*pro_pivot[-1].leave_force <pro_pivot[-1].prev2_force:
            return True,pro_pivot[-1].zg #buyback and suppor price
        else:
            return False,0  
        
def sell_point2(pro_pivot,tails,num_pivot=2):
    if len(pro_pivot)<=3 or tails[4]==-1 or pro_pivot[-1].size>=8 or pro_pivot[-1].finished!=0\
    or df1.iloc[-1][1]/pro_pivot[-1].leave_end_price -1<0 or \
     df1.iloc[-1][0] < tails[3]:
        return False,0
    else:#two pivot descending
        #no slow down
        if ( pro_pivot[-1].prev2_end_price >pro_pivot[-1].leave_end_price ) and \
        (pro_pivot[-1].leave_start_time==tails[0]) and\
             df1.iloc[-1][0] > 0.51*(pro_pivot[-1].zd+pro_pivot[-1].zg) and \
            pro_pivot[-1].prev2_end_price==pro_pivot[-1].gg:
            return True,pro_pivot[-1].zg #buyback and support price
        else:
            return False,0  
        
def sell_point3_ris(pro_pivot,tails,num_pivot=2):
    if len(pro_pivot)<=3 or tails[4]==-1 or pro_pivot[-1].size>=8 or pro_pivot[-1].finished!=1 \
    or \
     df1.iloc[-1][0] < tails[3]:
        return False,0
    else:#two pivot descending
        #no slow down
        if ( 1.02*pro_pivot[-1].leave_end_price < df1.iloc[-1][0] ) and \
        (pro_pivot[-1].leave_end_time==tails[0]) and \
            pro_pivot[-1].leave_force>pro_pivot[-1].prev2_force\
            and df1.iloc[-1][1]<pro_pivot[-1].zd:
            return True,pro_pivot[-1].zd # resistance price
        else:
            return False,0  
 
def main():
    df=pd.read_csv('C:/Users/Administrator/Desktop/ecom/chanlun/sh.csv',index_col=0)[['low','high']]
    df['datetime']=df.index
    #REMOVE INCLUSION
    while ( True ):
        temp_len = len(df)
        i=0
        while i<=len(df)-4:
            if (df.iloc[i+2,0]>=df.iloc[i+1,0] and df.iloc[i+2,1]<=df.iloc[i+1,1]) or\
            (df.iloc[i+2,0]<=df.iloc[i+1,0] and df.iloc[i+2,1]>=df.iloc[i+1,1]):
                if df.iloc[i+1,0]>df.iloc[i,0]:
                    df.iloc[i+2,0] = max(df.iloc[i+1:i+3,0])
                    df.iloc[i+2,1] = max(df.iloc[i+1:i+3,1])
                    df.drop(df.index[i+1],inplace=True)
                    
                    continue
                else:
                    df.iloc[i+2,0] = min(df.iloc[i+1:i+3,0])
                    df.iloc[i+2,1] = min(df.iloc[i+1:i+3,1])
                    df.drop(df.index[i+1],inplace=True)
                    
                    continue
            i = i + 1
       # print(len(df))    
        if len(df)==temp_len:
            break
            
    df= df.reset_index(drop=True)  
    #get difenxing and dingfenxing
    ul=[0]
    for i in range(len(df)-2):
        if df.iloc[i+2,0] < df.iloc[i+1,0] and df.iloc[i,0] < df.iloc[i+1,0]:
            ul = ul + [1]
            continue
        if df.iloc[i+2,0] > df.iloc[i+1,0] and df.iloc[i,0] > df.iloc[i+1,0]:
            ul = ul + [-1]# difenxing -1 dingfenxing +1
            continue
        else:
            ul = ul + [0]
    ul = ul + [0]
    global df1
    df1 = pd.concat((df[['low','high']],pd.DataFrame(ul),df['datetime']),axis=1)
      
    i = 0
    
    while df1.iloc[i,2] == 0 and i < len(df1)-2:
        i = i + 1
    df1=df1[i:]  
    
    i = 0
    while ( sum(abs(df1.iloc[i+1:i+4,2]))>0 or df1.iloc[i,2]==0) and i < len(df1)-2:
        i = i + 1
    df1=df1[i:]
    df1.rename(columns= {0:'od'},inplace=True)
    #df1.columns=Index(['low', 'high', 'od', 'datetime'], dtype='object')
    if len(df1)<=60:
        print('error!')
        return ;
    #remove those within 3 bars
    df1=df1.reset_index(drop=True)
    global od_list#od_list are the index of df1 whose corresponding point are fenxing extreme vertex
    
    od_list=[0]
    judge(0,0,1) 
    #od_list are the index of df1 whose corresponding point are fenxing extreme vertex
    #od_list are the index of df1 whose corresponding point are fenxing extreme vertex
    #od_list are the index of df1 whose corresponding point are fenxing extreme vertex
    #od_list are the index of df1 whose corresponding point are fenxing extreme vertex
    #od_list are the index of df1 whose corresponding point are fenxing extreme vertex
    #od_list are the index of df1 whose corresponding point are fenxing extreme vertex
    
    
    #generate seg
    start = 0
    while start < len(od_list)-5:
        if check_init_seg(od_list[start:start+4]):
            break
        else:
            start = start + 1
    
    lines = []
    
    i = start
    end = False
    while i <= len(od_list)-4:
        se = Seg(od_list[i:i+4])
        label = False
        while label == False and i <= len(od_list)-6:
            i = i + 2
            label,start = se.grow(od_list[i+2:i+4])
            if se.vertex[-1] > od_list[-3]:
                end =True
                
                lines += [se.lines()]
                break
        if end:
            break
        i =  np.where(np.array(od_list) == se.vertex[-1])[0][0]
        #show datetime of the end of the segment
        #print(df1.iloc[se.vertex[-1],3])   
        lines += [se.lines()]#there are still remaining fewer than or equal to 
        #3 bies not considered in the last
        #seg ,which is unfinished and named by tails
    low_list=df1.iloc[se.vertex[-1]:,0]
    high_list=df1.iloc[se.vertex[-1]:,1]
    
    low_extre=low_list.min()
    high_extre=high_list.max()
    if se.finished == True:
        if lines[-1][0][1] < lines[-1][1][1] :#d==1
            lines += [ [(se.vertex[-1],lines[-1][1][1]),(low_list.idxmin(),low_extre)]]
        else:
            lines += [ [(se.vertex[-1],lines[-1][1][1]),(high_list.idxmax(),high_extre)]]
            
    else:
        if lines[-1][0][1] < lines[-1][1][1] :#d==1
            if low_extre > lines[-1][0][1]:
                lines[-1] = [ (lines[-1][0][0],lines[-1][0][1]),(high_list.idxmax(),high_extre)] 
            else:
                if low_list.idxmin()-se.vertex[-1]>=10:
                    lines += [ [(se.vertex[-1],lines[-1][1][1]),(low_list.idxmin(),low_extre)]]
        else:
            
            if high_extre < lines[-1][0][1]:
                lines[-1] = [ (lines[-1][0][0],lines[-1][0][1]),(low_list.idxmin(),low_extre) ]
            else:
                if  high_list.idxmax()-se.vertex[-1]>=10:
                    lines += [ [(se.vertex[-1],lines[-1][1][1]),(high_list.idxmax(),high_extre)]]   
    
    #print(lines)
    #tails is the unfinished seg,tails[4] is its direction
    #lines are the segment,lines[0] is the first segment's start_index,start_price,end_index,end_price
    a,tails = get_pivot(lines)    
    pro_a= process_pivot(a)
    for i in range(len(pro_a)):
        print('pivot {}\'s info:'.format(i))
        pro_a[i].display()
    print('these index are for df1,not df!,ex:pro_a[0].leave_end_time to obtain the leave end time')
    
     
    
if __name__=="__main__": 
    main()
