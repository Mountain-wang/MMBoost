import numpy as np
import random as rd
from sklearn import tree
from numpy import dtype, vstack
#这是我根据论文手动编写的代码，因为找论文原作者要不到源代码


class UswitchingNEDclassifier:
    def __init__(self,numit=41,fr=0.1):
        self.numit=numit
        self.fr=fr
        pass
    
    def nedProb(self,pos_array,neg_array):
        negnum=len(neg_array)
        posnum=len(pos_array)
        #print('posnum:',posnum)
        if posnum==0:
            print('正样本个数为零，无法计算')
            exit(0)
        n2pmindist=[]
        sum1=0
        for i in range(negnum):
            mindist=-1
            for j in range(posnum):
                #print(j)
                dist1=np.sum((pos_array[j]-neg_array[i])**2)
                if mindist>dist1:
                    mindist=dist1
                if mindist==-1:
                    mindist=dist1
            n2pmindist.append((i,mindist))
            sum1=sum1+mindist
        n2pmindist.sort(key=lambda x:x[1], reverse=False) 
        nedprob=[]
        if sum1==0:
            print('除数不能为0')
            exit(0)
        starprob=0
        for x in n2pmindist:
            starprob=starprob+x[1]/sum1
            nedprob.append((x[0],starprob))
        return nedprob
    
    def n2pswitch(self,pos_array,neg_array,switchnum,nedprob):
        n2pidx_list=[]
        n2pidx_set=set()
        for i in range(switchnum):
            rdpb=rd.random()
            for x in nedprob:
                if rdpb<x[1]:
                    n2pidx_list.append(x[0])
                    n2pidx_set.add(x[0])
                    break
        n2p_array=neg_array[[x for x in n2pidx_list],:]
        pos_array=np.append(pos_array,n2p_array,axis=0)
        neg_array=np.delete(neg_array,[x for x in n2pidx_set],axis=0)
        return pos_array,neg_array
    
    def samplefromMaj(self,neg_array,samplenum):
        neglist=range(len(neg_array))
        sample_list=rd.sample(neglist,samplenum)
        return neg_array[sample_list]
        
    
    def fit(self,x_train,y_trainlabel):
        postlist=[]
        neglist=[]
        for i in range(len(y_trainlabel)):
            if y_trainlabel[i]==1:
                postlist.append(i)
            if y_trainlabel[i]==-1:
                neglist.append(i)
        pos_array, neg_array=x_train[postlist,:],x_train[neglist,:]     
        self.clfs=[]
        nedprob=self.nedProb(pos_array, neg_array)
        posnum=len(pos_array)
        negnum=len(neg_array)
        num=posnum+negnum
        pp=posnum/num
        pn=negnum/num
        switchnum=0
        samplenum=0
        if self.fr<(pn-pp)/2:
            newpp=0.5-self.fr
            switchnum=self.fr*num*pp/newpp
            switchnum=round(switchnum)
            samplenum=num*pn-(num*pp+switchnum*2)
            samplenum=round(samplenum)
        else:
            switchnum=int(num*self.fr)
            samplenum=0
        for it in range(self.numit):
            #pos_array1, neg_array1=pos_array.copy(), neg_array.copy()
            pos_train,neg_train=self.n2pswitch(pos_array, neg_array, switchnum, nedprob)
            if samplenum>0:
                neg_train=self.samplefromMaj(neg_train, samplenum)
            x_train=np.append(pos_train,neg_train,axis=0)
            y_trainlabel=[1]*len(pos_train)+[-1]*len(neg_train)
            #clf=tree.DecisionTreeClassifier(criterion='entropy')
            clf=tree.DecisionTreeClassifier()
            clf.fit(x_train,y_trainlabel)
            self.clfs.append(clf)
            
        return self
    def predict(self,x_test):
        res=np.zeros(len(x_test),dtype=np.int)
        if len(self.clfs)<self.numit:
            print('训练器数目不对')
            exit(0)
        for clf in self.clfs:
            res=res+clf.predict(x_test)
        res=np.sign(res)
        return res
    
    def predict_proba(self,x_test):
        if len(self.clfs)<self.numit:
            print('训练器数目不对')
            exit(0)
        res=np.zeros((len(x_test),2))
        res1=np.zeros(len(x_test))
        for clf in self.clfs:
            res1=res1+clf.predict(x_test)
        res1=np.sign(res1)
        res1=(res1+self.numit)/2
        res1proba=res1/self.numit
        res2proba=1-res1proba
        res=np.vstack((res2proba,res1proba))
        res=res.T
        return res
        
            