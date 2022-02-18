import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score,confusion_matrix,roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC,LinearSVC
from myadaboost import *
from imblearn.ensemble import EasyEnsembleClassifier
from datasetdeal import *
from ensemblemethods import *
import time

def neg2posInOneTime(pos_array,neg_array,):
    print('it is going')
    posarray_list=[]
    negarray_list=[]

    posarray_list.append(pos_array)
    negarray_list.append(neg_array)
    pos_num=len(pos_array)
    neg_num=len(neg_array)
    n=len(neg_array)/len(pos_array)
    n=round((n-1)/2)
    #if n>20:
        #n=20
    neg2pos_set=set()
    
    i=0
    while i<pos_num:#这种取最近的方式是不是合理的？可以接着探讨
        mindistance=1000000000
        neg2posindex=0
        for j in range(neg_num):
            if j in neg2pos_set:
                continue
            distance_ij=np.sum((pos_array[i]-neg_array[j])**2)   
            if distance_ij<mindistance:
                mindistance=distance_ij
                neg2posindex=j
        neg2pos_set.add(neg2posindex)
        i=i+1
        if i>=pos_num:
            neg2pso_array=neg_array[[x for x in neg2pos_set],:]
            leftneg_array=neg_array[[x for x in range(len(neg_array)) if x not in neg2pos_set],:]
            newpos_array=np.append(neg2pso_array,pos_array,axis=0)
            posarray_list.append(newpos_array)
            negarray_list.append(leftneg_array)
            n=n-1
            if n>0:
                i=0
    #已经包含了不做改变的数据，放在索引为0的地方
    return posarray_list,negarray_list

def lookup_n_InOneTime(pos_train,pos_test,neg_train,neg_test,clfmodel,k=1):
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    clf_list=[]
    n2p_index=0
    posarray_list,negarray_list=neg2posInOneTime(pos_train, neg_train)
    for eachpos_train,eachneg_train in zip(posarray_list,negarray_list):
        each_train=np.append(eachpos_train,eachneg_train,axis=0)
        each_label=[1]*len(eachpos_train)+[-1]*len(eachneg_train)
        if clfmodel=='cart':
            eachclf=tree.DecisionTreeClassifier(random_state=0)
        elif clfmodel=='rbf':
            eachclf=SVC(kernel='rbf',probability=True)
        elif clfmodel=='linear':
            eachclf=LinearSVC()
        elif clfmodel=='ada':
            eachclf=AdaBoostClassifier(n_estimators=40, random_state=0,algorithm='SAMME')
        elif clfmodel=='myada':
            eachclf=Myadaboost(n_estimators=40, random_state=0)
        elif clfmodel=='n2pee':
            adaclf=AdaBoostClassifier(n_estimators=10,algorithm='SAMME')
            eachclf=EasyEnsembleClassifier(random_state=0,n_estimators=4,base_estimator=adaclf)
        elif clfmodel=='n2p':
            eachclf=N2padaboost(n_estimators=40, random_state=0)
            #n2p_index=n2p_index+len(pos_train)
            eachclf.fit(each_train,each_label,n2p_index,len(pos_train))
            n2p_index=n2p_index+len(pos_train)
            clf_list.append(eachclf)
            continue
        elif clfmodel=='bs':
            eachclf=Stumpadaboost()
            
            #n2p_index=n2p_index+len(pos_train)
            eachclf.fit(each_train,each_label,k,t=n2p_index)
            n2p_index=n2p_index+len(pos_train)
            clf_list.append(eachclf)
            continue
        else:
            print('mis_input_clf')
            return 0
        #cbuInTrain(each_train, each_label, x_test, y_testlabel, clfmodel)
        eachclf.fit(each_train,each_label)
        clf_list.append(eachclf)
    '添加一个list记录结果列表'
    predict_list=[]
    predict_list.append(y_testlabel)
    allscore_list=[]
    for j in range(len(clf_list)):
        eachclf=clf_list[j]
        eachclf_predict=eachclf.predict(x_test)
        eachclf_predict_proba=eachclf.predict_proba(x_test)
        predict_list.append(eachclf_predict)#记录本次预测的结果
        conmat=confusion_matrix(y_testlabel,eachclf_predict)
        tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
        tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
        g_mean=(tpr*tnr)**0.5
        f1score=f1_score(y_testlabel, eachclf_predict)
        aucscore=roc_auc_score(y_testlabel,eachclf_predict_proba[:,1])
        #=======================================================================
        # print('n=',j,'情况下混淆矩阵:','\n',confusion_matrix(y_testlabel, eachclf_predict))
        # print('n=',j,'情况下f1:',f1_score(y_testlabel, eachclf_predict))
        # print('n=',j,'情况下G-MEAN:',g_mean)
        # print('n=',j,'情况下整体精确度:',accuracy_score(y_testlabel, eachclf_predict))
        # print('n=',j,'情况下模型AUC:',aucscore)
        #=======================================================================
        score_list=[]
        score_list.append(f1_score(y_testlabel, eachclf_predict))
        score_list.append(g_mean)
        score_list.append(accuracy_score(y_testlabel, eachclf_predict))
        score_list.append(aucscore)
        #allscore_list.append(score_list)
        #为了最后的输出，不采用append，而采用extend
        allscore_list.extend(score_list)
    #===========================================================================
    # #将结果写入目标文件
    # predict_array=np.array(predict_list)
    # predict_array=predict_array.T
    # name=['truelabel']+['clf'+str(x) for x in range(len(clf_list))]
    # predict_data=pd.DataFrame(columns=name,data=predict_array)
    # #filepath=r'D:\Codetool\datasrc\imbalancedata\abalone\testclfpre.csv'
    # filepath=r'D:\Codetool\datasrc\imbalancedata\mf-morph\mfmtestclfpre.csv'
    # predict_data.to_csv(filepath)
    #===========================================================================
    
    return allscore_list 

def glassdataForTest(dataset_num):
    posarray,negarray,resultpath,k=get33Data(dataset_num)
    #k=2*k
    clfmodel='bs'
    #score_list=[]
    kf=KFold(5,shuffle=True)
    allkfscore_list=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        eachscore_list=[]
        russcore_list=rusboost_GetScore(pos_train, pos_test, neg_train, neg_test)
        print('rus is ok')
        easyscore_list=easyensenmble_GetScore(pos_train, pos_test, neg_train, neg_test)
        babagscore_list=balancebagging_GetScore(pos_train, pos_test, neg_train, neg_test)
        baforestscore_list=balancerandomforest_GetScore(pos_train, pos_test, neg_train, neg_test)
        cbuscore_list=cbuInTrain_score(pos_train, pos_test, neg_train, neg_test)
        #balancebaggingSmote_GetScore
        #cbuscore_list=balancebaggingSmote_GetScore(pos_train, pos_test, neg_train, neg_test)#这里其实是smotebag
        looknscore_list=lookup_n_InOneTime(pos_train, pos_test, neg_train, neg_test, clfmodel, k)
        eachscore_list.extend(russcore_list)
        eachscore_list.extend(easyscore_list)
        eachscore_list.extend(babagscore_list)
        eachscore_list.extend(baforestscore_list)
        eachscore_list.extend(cbuscore_list)
        eachscore_list.extend(looknscore_list)#这里用的是extend
        #extend：l1=[1,2],l1=[3,4],则l1.extend(l2)=[1,2,3,4]
        #而append是[[1,2],[3,4]]
        allkfscore_list.append(eachscore_list)#这里用的是append
    #===========================================================================
    # data_array=np.array(allkfscore_list)
    # data=pd.DataFrame(data_array)
    # firstline_str= time.asctime( time.localtime(time.time()) )
    # firstline_str=firstline_str+'  mymethod take the '+clfmodel
    # fline=[[firstline_str],[firstline_str]]
    # fline_array=np.array(fline)
    # data_str=pd.DataFrame(fline_array)
    # path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\result.csv'  
    # data_str.to_csv(path,mode='a',header = False, index = False)
    # data.to_csv(path,mode='a',header = False, index = False)
    #===========================================================================
    #path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\result.csv' 
    savetheresult(allkfscore_list, resultpath, clfmodel,k)#将上面注释的直接包装成函数
def addZero(aslist):
    aslen=len(aslist)
    maxl=len(aslist[0])
    for i in range(aslen):
        if len(aslist[i])>maxl:
            maxl=len(aslist[i])
    for i in range(aslen):
        zeronum=maxl-len(aslist[i])
        if zeronum>0:
            zerolist=[0]*zeronum
            aslist[i].extend(zerolist)
        pass
    return aslist
    
def savetheresult(allkfscore_list,path,clfmodel,k):
    allkfscore_list=addZero(allkfscore_list)
    data_array=np.array(allkfscore_list)
    print(data_array.shape)
    data=pd.DataFrame(data_array)
    firstline_str= time.asctime( time.localtime(time.time()) )
    firstline_str=firstline_str+'  mymethod take the '+clfmodel+'  k='+str(k)
    fline=[[firstline_str],[firstline_str]]
    fline_array=np.array(fline)
    data_str=pd.DataFrame(fline_array)
    #path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\result.csv'  
    data_str.to_csv(path,mode='a',header = False, index = False)
    data.to_csv(path,mode='a',header = False, index = False)

def oneOrAllTest():
    ot=input('请问是单独测试某个数据集还是所有（1--单独，2---所有(0-31)）')
    if ot=='1':
        dataset_num=input('请输入使用的第几个数据集，从0--32选择')
        glassdataForTest(int(dataset_num))
    if ot=='2':
        print('即将进行所有所有数据集的运行，时间可能比较久')
        for i in range(32):
            glassdataForTest(i)
        

if __name__=='__main__':
    oneOrAllTest()
