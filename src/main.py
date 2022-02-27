import numpy as np
import pandas as pd
from datasetdeal import *
from ensemblemethods import *
from buchongshiyan import *
from sklearn.model_selection._split import KFold
import time

if  __name__=='__main__':
    path=input('Please input the path of dataset (.csv)')
    posarray,negarray=getTargetDdata(path)
    r,k=input('please input the parameters R,K,the default is R=1,K=1')
    r=int(r)
    k=float(k)
    kf=KFold(5,shuffle=True)
    fivetimescore=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        n2pidx=0+len(pos_train)*r
        pos_train,neg_train=n2pwithR(pos_train, neg_train, r)
        each_cartscore=mmboost_GetScore(pos_train,pos_test,neg_train,neg_test,k,n2pidx)
        fivetimescore.append(each_cartscore)
    np.array(fivetimescore)
    print(["f1","G-mean","ACC","AUC"])
    print(fivetimescore)
    
    
    