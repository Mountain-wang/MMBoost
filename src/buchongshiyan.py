#补充实验，1、补上基线实验的结果。2、补上其他几个实验结果

import numpy as np
import pandas as pd
from datasetdeal import *
from ensemblemethods import *
from sklearn.model_selection._split import KFold
import time

def n2pwithR(pos_array,neg_array,r):
    n2pidx_set=set()
    posnum=len(pos_array)
    negnum=len(neg_array)
    for rn in range(r):
        for i in range(posnum):
            mindist=-1
            minidx=0
            for j in range(negnum):
                if j in n2pidx_set:
                    continue
                tmdist=np.sum((pos_array[i]-neg_array[j])**2)#这是采用欧式距离
                #tmdist=np.linalg.norm(pos_array[i]-neg_array[j],ord=1)#这是曼哈顿距离
                #tmdist=np.dot(pos_array[i],neg_array[j])/(np.linalg.norm(pos_array)*np.linalg.norm(neg_array))
                if mindist==-1:
                    mindist=tmdist
                    minidx=j
                if tmdist<mindist:
                    mindist=tmdist
                    minidx=j
            n2pidx_set.add(minidx)
    n2p_array=neg_array[[x for x in n2pidx_set],:]
    pos_array=np.append(pos_array,n2p_array,axis=0)
    neg_array=np.delete(neg_array,[x for x in n2pidx_set],axis=0)
    return pos_array,neg_array
        
def mmboostFourScoreSignle(data_num):      
    posarray,negarray,_,k,r=getRKon33Data(data_num)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        n2pidx=0+len(pos_train)*r
        pos_train,neg_train=n2pwithR(pos_train, neg_train, r)
        each_cartscore=mmboost_GetScore(pos_train,pos_test,neg_train,neg_test,k,n2pidx)
        fivetimescore.append(each_cartscore)
    return np.array(fivetimescore)

def mmboostaddFourScoreAlldataset():
    allscore=mmboostFourScoreSignle(0)
    for i in range(1,33):
        temp=mmboostFourScoreSignle(i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\secondex\mmboost.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the mmboost'
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print('mmboost have done')

            
            
 
def cartaddFourScoreSignle(data_num):
    posarray,negarray,_,_=get33Data(data_num)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        each_cartscore=cart_GetScore(pos_train, pos_test, neg_train, neg_test)
        fivetimescore.append(each_cartscore)
    return np.array(fivetimescore)#返回值是ndarray类型，不是list

def cartaddFourScoreAlldataset():
    allscore=cartaddFourScoreSignle(0)
    for i in range(1,33):
        temp=cartaddFourScoreSignle(i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\secondex\cart.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the cart'
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print('cart have done')

def smotebagaddFourScoreSignle(data_num):
    posarray,negarray,_,_=get33Data(data_num)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        each_cartscore=balancebaggingSmote_GetScore(pos_train, pos_test, neg_train, neg_test)
        #each_cartscore=balancerandomforest_GetScore(pos_train,pos_test,neg_train,neg_test)#这个是可以的
        #each_cartscore=balancebagging_GetScore(pos_train, pos_test, neg_train, neg_test)
        fivetimescore.append(each_cartscore)
    return np.array(fivetimescore)#返回值是ndarray类型，不是list

def smotebagAddFourScoreAlldataset():
    allscore=smotebagaddFourScoreSignle(0)
    for i in range(1,33):
        if i==14 or i==18 or i==22 or i==23 or i==25 or i==30:
            temp_score=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
            allscore=np.hstack((allscore,temp_score))
            continue
        temp=smotebagaddFourScoreSignle(i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\secondex\smotebag.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the smotebag'
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print('smotebag have done')

def rusFourScoreSignle(data_num):
    posarray,negarray,_,_=get33Data(data_num)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        #print(type(pos_test))
        each_cartscore=rusboost_GetScore(pos_train, pos_test, neg_train, neg_test)
        fivetimescore.append(each_cartscore)
    return np.array(fivetimescore)#返回值是ndarray类型，不是list

def rusAddFourScoreAlldataset():
    allscore=rusFourScoreSignle(0)
    for i in range(1,33):
        temp=rusFourScoreSignle(i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\secondex\rus.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the rus'
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print('rus have done')

def easyenFourScoreSignle(data_num):
    posarray,negarray,_,_=get33Data(data_num)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        each_cartscore=easyensenmble_GetScore(pos_train, pos_test, neg_train, neg_test)
        fivetimescore.append(each_cartscore)
    return np.array(fivetimescore)#返回值是ndarray类型，不是list

def easyenAddFourScoreAlldataset():
    allscore=easyenFourScoreSignle(0)
    for i in range(1,33):
        temp=easyenFourScoreSignle(i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\secondex\easy.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the easyen'
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print('easy have done')

def ubagFourScoreSignle(data_num):
    posarray,negarray,_,_=get33Data(data_num)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        each_cartscore=balancebagging_GetScore(pos_train, pos_test, neg_train, neg_test)
        fivetimescore.append(each_cartscore)
    return np.array(fivetimescore)#返回值是ndarray类型，不是list 

def ubagAddFourScoreAlldataset():
    allscore=ubagFourScoreSignle(0)
    for i in range(1,33):
        temp=ubagFourScoreSignle(i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\secondex\ubag.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the ubag'
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print('ubag have done')

def brfFourScoreSignle(data_num):
    posarray,negarray,_,_=get33Data(data_num)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        each_cartscore=balancerandomforest_GetScore(pos_train, pos_test, neg_train, neg_test)
        fivetimescore.append(each_cartscore)
    return np.array(fivetimescore)#返回值是ndarray类型，不是list 

def brfAddFourScoreAlldataset():
    allscore=brfFourScoreSignle(0)
    for i in range(1,33):
        temp=brfFourScoreSignle(i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\secondex\brf.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the brf'
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print('brf have done')

def cbuFourScoreSignle(data_num):
    posarray,negarray,_,_=get33Data(data_num)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        each_cartscore=cbuInTrain_score(pos_train, pos_test, neg_train, neg_test)
        fivetimescore.append(each_cartscore)
    return np.array(fivetimescore)#返回值是ndarray类型，不是list 

def cbuAddFourScoreAlldataset():
    allscore=cbuFourScoreSignle(0)
    for i in range(1,33):
        temp=cbuFourScoreSignle(i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\secondex\cbu.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the cbu'
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print('cbu have done')

def usendFourScoreSignle(data_num):
    posarray,negarray,_,_=get33Data(data_num)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        each_cartscore=usend_GetScore(pos_train, pos_test, neg_train, neg_test)
        #print(each_cartscore)
        fivetimescore.append(each_cartscore)
    return np.array(fivetimescore)#返回值是ndarray类型，不是list 

def usendAddFourScoreAlldataset():
    allscore=usendFourScoreSignle(0)
    for i in range(1,33):
        temp=usendFourScoreSignle(i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\secondex\usend.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the usend'
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print('usend have done')

def smoteboostFourScoreSignle(data_num):
    posarray,negarray,_,_=get33Data(data_num)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        each_cartscore=smoboost_GetScore(pos_train, pos_test, neg_train, neg_test)
        #print(each_cartscore)
        fivetimescore.append(each_cartscore)
    return np.array(fivetimescore)#返回值是ndarray类型，不是list 

def smoteboostdAddFourScoreAlldataset():
    allscore=smoteboostFourScoreSignle(0)
    for i in range(1,33):
        if i==23 or i==30:
            temp_score=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
            allscore=np.hstack((allscore,temp_score))
            continue
        temp=smoteboostFourScoreSignle(i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\secondex\smboost.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the smboost'
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print('smoost have done')

def choseModelSignle(ss,data_num):
    if ss=="brf":
        return brfFourScoreSignle(data_num)
    elif ss=="cart":
        return cartaddFourScoreSignle(data_num)
    elif ss=="mmb":
        return mmboostFourScoreSignle(data_num)
    elif ss=="usend":
        return usendFourScoreSignle(data_num)
    elif ss=="easy":
        return easyenFourScoreSignle(data_num)
    elif ss=="ubag":
        return ubagFourScoreSignle(data_num)
    elif ss=="sbo":
        return smoteboostFourScoreSignle(data_num)
    elif ss=="sba":
        return smotebagaddFourScoreSignle(data_num)
    elif ss=="rus":
        return rusFourScoreSignle(data_num)
    else:
        print("没有此模型")
        exit(0)
    return []

def addResultOnTenSet(ss):
    allscore=choseModelSignle(ss, 33)
    for i in range(34,43):
        if i==37 or i==41:
            if ss=="usend" or ss=="mmb":
                temp_score=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
                allscore=np.hstack((allscore,temp_score))
                continue
        temp=choseModelSignle(ss, i)
        allscore=np.hstack((allscore,temp))
        
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\thirdex\\'+ss+'.csv'
    
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the'+ss
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    data_str.to_csv(path,mode='a',header = False, index = False)
    
    dataf_allscore=pd.DataFrame(allscore)
    dataf_allscore.to_csv(path,mode='a',header = False, index = False)
    print(ss+' have done')
    pass

if __name__=='__main__':

    #cartaddFourScoreAlldataset()
    #smotebagAddFourScoreAlldataset()
    #rusAddFourScoreAlldataset()   
    #easyenAddFourScoreAlldataset()
    #ubagAddFourScoreAlldataset()
    #brfAddFourScoreAlldataset()
    #cbuAddFourScoreAlldataset()
    #usendAddFourScoreAlldataset()
    #smoteboostdAddFourScoreAlldataset()
    #mmboostaddFourScoreAlldataset()
    
    addResultOnTenSet("mmb")