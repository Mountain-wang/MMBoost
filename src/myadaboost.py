from sklearn.tree import DecisionTreeClassifier
import numpy as np
import math
from math import *
from cmath import inf


class N2padaboost:#只使用决策树
    def __init__(self,n_estimators=10,random_state=0):
        self.n_estimators=n_estimators
        self.random_state=random_state
    
    def fit(self,x_train,y_trainlabel,n2p_index,pos_num):   
        self.clf_list=[]
        train_num=x_train.shape[0]
        #改变初始化权重       
        weight=[0]*train_num
        ir=(train_num/pos_num)-1
        k=n2p_index/pos_num
        for i in range(train_num):
            if i<n2p_index+pos_num:
                weight[i]=1/((k+1)*2*pos_num)
            else:
                weight[i]=1/((ir-k)*2*pos_num)
        #weight=np.ones(train_num)/train_num
        self.alpha=[] 
        current_pre=np.zeros(train_num)
        for j in range(self.n_estimators): 
            eachclf=DecisionTreeClassifier(random_state=self.random_state,max_depth=2)
            eachclf.fit(x_train,y_trainlabel,sample_weight=weight)
            self.clf_list.append(eachclf)
            each_predict=eachclf.predict(x_train)
            #a=[int(x!=y) for (x,y) in zip(y_trainlabel,y_truelabel)]
            #n2p_index=np.sum(a)
            #miss=[int(x!=y) for (x,y) in zip(each_predict[n2p_index:],y_trainlabel[n2p_index:])]
            miss=[int(x!=y) for (x,y) in zip(each_predict,y_trainlabel)]
            error=np.dot(weight,miss)
            alpha_j=0.5*math.log((1-error)/error)
            self.alpha.append(alpha_j)
            current_pre=current_pre+np.dot(alpha_j,each_predict)
            pre_sign=np.sign(current_pre)
            y_truelabel=[-1]*n2p_index+y_trainlabel[n2p_index:]
            if (pre_sign==y_trainlabel).all():
                break
            if (pre_sign==y_truelabel).all:
                break
            
            
            for i in range(train_num):                   
                weight[i]=weight[i]*np.exp(-1*alpha_j*y_trainlabel[i]*each_predict[i])
                     
            for i in range(train_num):
                weight[i]/=np.sum(weight)
        
        return self   
    
    def predict(self,x_test):
        final_pre=np.zeros(len(x_test))
        for i in range(len(self.clf_list)):
            current_pre=self.clf_list[i].predict(x_test)
            final_pre=final_pre+np.dot(self.alpha[i],current_pre)
        
        y_predict=np.sign(final_pre)
            
        return y_predict

class Myadaboost:
    def __init__(self,n_estimators=40,random_state=0):
        self.n_estimators=n_estimators
        self.random_state=random_state
    
    def fit(self,x_train,y_trainlabel):   
        self.clf_list=[]
        train_num=x_train.shape[0]
        weight=np.ones(train_num)/train_num
        self.alpha=[] 
        current_pre=np.zeros(train_num)
        for j in range(self.n_estimators): 
            eachclf=DecisionTreeClassifier(random_state=self.random_state,max_depth=1)
            eachclf.fit(x_train,y_trainlabel,sample_weight=weight)
            self.clf_list.append(eachclf)
            each_predict=eachclf.predict(x_train)
            #a=[int(x!=y) for (x,y) in zip(y_trainlabel,y_truelabel)]
            #n2p_index=np.sum(a)
            miss=[int(x!=y) for (x,y) in zip(each_predict,y_trainlabel)]
            error=np.dot(weight,miss)
            alpha_j=0.5*math.log((1-error)/error)
            self.alpha.append(alpha_j)
            current_pre=current_pre+np.dot(alpha_j,each_predict)
            pre_sign=np.sign(current_pre)
            if (pre_sign==y_trainlabel).all():
                break
            
            
            for i in range(train_num):
                weight[i]=weight[i]*np.exp(-1*alpha_j*y_trainlabel[i]*each_predict[i])                     
            for i in range(train_num):
                weight[i]/=np.sum(weight)
        
        return self   
    
    def predict(self,x_test):
        final_pre=np.zeros(len(x_test))
        for i in range(len(self.clf_list)):
            current_pre=self.clf_list[i].predict(x_test)
            final_pre=final_pre+np.dot(self.alpha[i],current_pre)
        
        y_predict=np.sign(final_pre)
            
        return y_predict

class Stumpadaboost:#使用树桩做基分类器
    def __init__(self,numit=40):
        self.numit=numit
        self.weakclassarr=[]
        pass
    def stumpClassify(self,datamatrix,dimen,threshval,threshineq):
        retarray=np.ones((np.shape(datamatrix)[0],1))
        if threshineq=='lt':
            retarray[datamatrix[:,dimen]<=threshval]=-1.0
        else:
            retarray[datamatrix[:,dimen]>threshval]=-1.0
        return retarray
    def buildstump(self,dataarr,classlabels,D,k,t):#传进来的D是个行向量，只有一行
        datamatrix=np.mat(dataarr);labelmat=np.mat(classlabels).T
        m,n=np.shape(datamatrix)
        #有个重要的参数，分割次数
        numsteps=10.0;beststump={};bestclassest=np.mat(np.zeros((m,1)))
        minerror=inf
        bestonesrror=0
        besttwoerror=0
        for i in range(n):
            rangemin=datamatrix[:,i].min()
            rangemax=datamatrix[:,i].max()
            stepsize=(rangemax-rangemin)/numsteps
            for j in range(-1,int(numsteps)+1):
                for inequal in ['lt','gt']:
                    threshval=(rangemin+float(j)*stepsize)
                    predictedvals=self.stumpClassify(datamatrix,i,threshval,inequal)
                    errarr=np.mat(np.ones((m,1)))
                    errarr[predictedvals==labelmat]=0
                    oneerror=D.T*errarr
                    if t==0:
                        twoerror=0
                    else:
                        twoerror=D[:,:t].T*errarr[:,:t]
                    weightederror=oneerror+(k-1)*twoerror
                    #print('特征',i,'阈值',threshval,'取向',inequal,'误差',weightederror)
                    if weightederror<minerror:
                        minerror=weightederror
                        bestclassest=predictedvals.copy()
                        bestonesrror=oneerror
                        besttwoerror=twoerror
                        beststump['dim']=i
                        beststump['thresh']=threshval
                        beststump['ineq']=inequal
        
        return beststump,bestonesrror,besttwoerror,bestclassest
    
    def fit(self,dataarr,classlabels,k,t):
        #print('k的值是:',k)
        #self.weakclassarr=[]
        m=np.shape(dataarr)[0]
        D=np.mat(np.ones((m,1))/m)
        aggclassest=np.mat(np.zeros((m,1)))
        for i in range(self.numit):
            beststump,em,et,classest=self.buildstump(dataarr, classlabels, D, k,t)
            wt=D[:t,:].sum()
            a1=(1-em)+(k-1)*(wt-et)
            a2=em+(k-1)*et
            alpha=float(0.5)*log(max(a1,1e-16)/max(a2,1e-16))
            
            beststump['alpha']=alpha
            #xx={'alpha':alpha}
            #beststump.update(xx)
            self.weakclassarr.append(beststump)
            #print(self.weakclassarr)
            expon=np.multiply(-1*alpha*np.mat(classlabels).T,classest)
            D=np.multiply(D,np.exp(expon))
            D=D/D.sum()
            aggclassest+=alpha*classest
            aggerrors=np.multiply(np.sign(aggclassest)!=np.mat(classlabels).T,np.ones((m,1)))
            errorrate=aggerrors.sum()/m
            if errorrate==0:
                break
        return self,self.weakclassarr
    
    def predict(self,x_test):
        #print(self.weakclassarr)
        datamatrix=np.mat(x_test)
        m=np.shape(datamatrix)[0]
        final_pre=np.zeros((m,1))
        for i in range(len(self.weakclassarr)):
            bestarg=self.weakclassarr[i]#一个词典
            bestdim=bestarg['dim']
            bestthr=bestarg['thresh']
            bestineq=bestarg['ineq']
            alpha_i=bestarg['alpha']
            current_pre=self.stumpClassify(datamatrix, bestdim, bestthr, bestineq)
            final_pre=final_pre+alpha_i*current_pre
        
        y_predict=np.sign(final_pre)
            
        return y_predict
    #这个函数是用来预测属于各个类别的概率的，参照其他的，也是两列，第一列是属于‘-1’的概率
    #第二列是属于‘1’的概率
    def predict_proba(self,x_test):
        datamatrix=np.mat(x_test)
        m=np.shape(datamatrix)[0]
        final_pre=np.zeros((m,1))
        for i in range(len(self.weakclassarr)):
            bestarg=self.weakclassarr[i]#一个词典
            bestdim=bestarg['dim']
            bestthr=bestarg['thresh']
            bestineq=bestarg['ineq']
            alpha_i=bestarg['alpha']
            current_pre=self.stumpClassify(datamatrix, bestdim, bestthr, bestineq)
            final_pre=final_pre+alpha_i*current_pre
        
        y_predict_proba=final_pre/2+0.5
        ti=1-y_predict_proba
        y_predict_proba=np.append(ti,y_predict_proba,axis=1)
            
        return y_predict_proba
                        
                    
                    
            
            
        