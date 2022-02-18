import numpy as np
from imblearn.ensemble import *
from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,accuracy_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from usend import *
from myadaboost import *

def rusboost_GetScore(pos_train,pos_test,neg_train,neg_test):
    x_train=np.append(pos_train,neg_train,axis=0)
    y_trainlabel=[1]*len(pos_train)+[-1]*len(neg_train)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    #clf=RUSBoostClassifier(random_state=0,n_estimators=40)
    #c1=UswitchingNEDclassifier()
    #clf=RUSBoostClassifier(base_estimator=c1,random_state=0,n_estimators=10)
    clf=RUSBoostClassifier(random_state=0,n_estimators=10)
    clf.fit(x_train,y_trainlabel)
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)#这个是带概率的结果
    auc=roc_auc_score(y_testlabel, y_pred_proba[:,1])
    conmat=confusion_matrix(y_testlabel, y_pred)
    tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
    tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
    g_mean=(tpr*tnr)**0.5
    acc=accuracy_score(y_testlabel, y_pred)
    f1=f1_score(y_testlabel, y_pred)
    clf_name='RUSBOOST'
    #===========================================================================
    # print('********',clf_name ,'method ******')
    # print(clf_name,' -confusionmatrix:','\n',conmat)
    # print(clf_name,' -F-measure:',f1)
    # print(clf_name,' -G-MEAN:',g_mean)
    # print(clf_name,' -整体精确度:',acc)
    # print(clf_name,' -AUC：',auc)
    #===========================================================================
    
    score_list=[]
    score_list.append(f1)
    score_list.append(g_mean)
    score_list.append(acc)
    score_list.append(auc)
    return score_list

def easyensenmble_GetScore(pos_train,pos_test,neg_train,neg_test):
    x_train=np.append(pos_train,neg_train,axis=0)
    y_trainlabel=[1]*len(pos_train)+[-1]*len(neg_train)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    clf=EasyEnsembleClassifier(random_state=0)
    clf.fit(x_train,y_trainlabel)
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)#这个是带概率的结果
    auc=roc_auc_score(y_testlabel, y_pred_proba[:,1])
    conmat=confusion_matrix(y_testlabel, y_pred)
    tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
    tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
    g_mean=(tpr*tnr)**0.5
    acc=accuracy_score(y_testlabel, y_pred)
    f1=f1_score(y_testlabel, y_pred)
    #===========================================================================
    # clf_name='EasyEnsemble'
    # print('********',clf_name ,'method ******')
    # print(clf_name,' -confusionmatrix:','\n',conmat)
    # print(clf_name,' -F-measure:',f1)
    # print(clf_name,' -G-MEAN:',g_mean)
    # print(clf_name,' -整体精确度:',acc)
    # print(clf_name,' -AUC：',auc)
    #===========================================================================
    
    score_list=[]
    score_list.append(f1)
    score_list.append(g_mean)
    score_list.append(acc)
    score_list.append(auc)
    return score_list

def balancebagging_GetScore(pos_train,pos_test,neg_train,neg_test):
    x_train=np.append(pos_train,neg_train,axis=0)
    y_trainlabel=[1]*len(pos_train)+[-1]*len(neg_train)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    clf=BalancedBaggingClassifier(random_state=0)
    clf.fit(x_train,y_trainlabel)
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)#这个是带概率的结果
    auc=roc_auc_score(y_testlabel, y_pred_proba[:,1])
    conmat=confusion_matrix(y_testlabel, y_pred)
    tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
    tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
    g_mean=(tpr*tnr)**0.5
    acc=accuracy_score(y_testlabel, y_pred)
    f1=f1_score(y_testlabel, y_pred)
    #===========================================================================
    # clf_name='BalancedBagging'
    # print('********',clf_name ,'method ******')
    # print(clf_name,' -confusionmatrix:','\n',conmat)
    # print(clf_name,' -F-measure:',f1)
    # print(clf_name,' -G-MEAN:',g_mean)
    # print(clf_name,' -整体精确度:',acc)
    # print(clf_name,' -AUC：',auc)
    #===========================================================================
    
    score_list=[]
    score_list.append(f1)
    score_list.append(g_mean)
    score_list.append(acc)
    score_list.append(auc)
    return score_list

def balancerandomforest_GetScore(pos_train,pos_test,neg_train,neg_test):
    x_train=np.append(pos_train,neg_train,axis=0)
    y_trainlabel=[1]*len(pos_train)+[-1]*len(neg_train)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    clf=BalancedRandomForestClassifier(max_depth=2, random_state=0,n_estimators=40)
    clf.fit(x_train,y_trainlabel)
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)#这个是带概率的结果
    auc=roc_auc_score(y_testlabel, y_pred_proba[:,1])
    conmat=confusion_matrix(y_testlabel, y_pred)
    tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
    tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
    g_mean=(tpr*tnr)**0.5
    acc=accuracy_score(y_testlabel, y_pred)
    f1=f1_score(y_testlabel, y_pred)
    #===========================================================================
    # clf_name='BalancedRandomForest'
    # print('********',clf_name ,'method ******')
    # print(clf_name,' -confusionmatrix:','\n',conmat)
    # print(clf_name,' -F-measure:',f1)
    # print(clf_name,' -G-MEAN:',g_mean)
    # print(clf_name,' -整体精确度:',acc)
    # print(clf_name,' -AUC：',auc)
    #===========================================================================
    
    score_list=[]
    score_list.append(f1)
    score_list.append(g_mean)
    score_list.append(acc)
    score_list.append(auc)
    return score_list


def cbuInTrain_score(pos_train,pos_test,neg_train,neg_test):
    #这个函数方法虽然不是集成，但是聚类下采样的一种方法
    x_train=np.append(pos_train,neg_train,axis=0)
    y_trainlabel=[1]*len(pos_train)+[-1]*len(neg_train)
    cc=ClusterCentroids(random_state=0)
    x_train,y_trainlabel=cc.fit_resample(x_train,y_trainlabel)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    clf=tree.DecisionTreeClassifier(random_state=0)
    clf.fit(x_train,y_trainlabel)
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)#这个是带概率的结果
    auc=roc_auc_score(y_testlabel, y_pred_proba[:,1])
    conmat=confusion_matrix(y_testlabel, y_pred)
    tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
    tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
    g_mean=(tpr*tnr)**0.5
    acc=accuracy_score(y_testlabel, y_pred)
    f1=f1_score(y_testlabel, y_pred)
    #===========================================================================
    # clf_name='cbuInTrain'
    # print('********',clf_name ,'method ******')
    # print(clf_name,' -confusionmatrix:','\n',conmat)
    # print(clf_name,' -F-measure:',f1)
    # print(clf_name,' -G-MEAN:',g_mean)
    # print(clf_name,' -整体精确度:',acc)
    # print(clf_name,' -AUC：',auc)
    #===========================================================================
    
    score_list=[]
    score_list.append(f1)
    score_list.append(g_mean)
    score_list.append(acc)
    score_list.append(auc)
    return score_list

def balancebaggingSmote_GetScore(pos_train,pos_test,neg_train,neg_test):
    x_train=np.append(pos_train,neg_train,axis=0)
    y_trainlabel=[1]*len(pos_train)+[-1]*len(neg_train)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    clf=BalancedBaggingClassifier(random_state=0,sampler=SMOTE())
    clf.fit(x_train,y_trainlabel)
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)#这个是带概率的结果
    auc=roc_auc_score(y_testlabel, y_pred_proba[:,1])
    conmat=confusion_matrix(y_testlabel, y_pred)
    tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
    tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
    g_mean=(tpr*tnr)**0.5
    acc=accuracy_score(y_testlabel, y_pred)
    f1=f1_score(y_testlabel, y_pred)
    #===========================================================================
    # clf_name='BalancedBagging'
    # print('********',clf_name ,'method ******')
    # print(clf_name,' -confusionmatrix:','\n',conmat)
    # print(clf_name,' -F-measure:',f1)
    # print(clf_name,' -G-MEAN:',g_mean)
    # print(clf_name,' -整体精确度:',acc)
    # print(clf_name,' -AUC：',auc)
    #===========================================================================
    
    score_list=[]
    score_list.append(f1)
    score_list.append(g_mean)
    score_list.append(acc)
    score_list.append(auc)
    return score_list

def cart_GetScore(pos_train,pos_test,neg_train,neg_test):#虽然论文采用的是C4.5,但CART是官方的，效果更好
    x_train=np.append(pos_train,neg_train,axis=0)
    y_trainlabel=[1]*len(pos_train)+[-1]*len(neg_train)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    clf=tree.DecisionTreeClassifier(random_state=0,criterion='entropy')
    clf.fit(x_train,y_trainlabel)
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)#这个是带概率的结果，因为AUC的计算要使用
    auc=roc_auc_score(y_testlabel, y_pred_proba[:,1])
    conmat=confusion_matrix(y_testlabel, y_pred)
    tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
    tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
    g_mean=(tpr*tnr)**0.5
    acc=accuracy_score(y_testlabel, y_pred)
    f1=f1_score(y_testlabel, y_pred)
    score_list=[]
    score_list.append(f1)
    score_list.append(g_mean)
    score_list.append(acc)
    score_list.append(auc)
    return score_list

def usend_GetScore(pos_train,pos_test,neg_train,neg_test):#虽然论文采用的是C4.5,但CART是官方的，效果更好
    x_train=np.append(pos_train,neg_train,axis=0)
    y_trainlabel=[1]*len(pos_train)+[-1]*len(neg_train)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    clf=UswitchingNEDclassifier()
    clf.fit(x_train,y_trainlabel)
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)#这个是带概率的结果，因为AUC的计算要使用
    auc=roc_auc_score(y_testlabel, y_pred_proba[:,1])
    conmat=confusion_matrix(y_testlabel, y_pred)
    tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
    tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
    g_mean=(tpr*tnr)**0.5
    acc=accuracy_score(y_testlabel, y_pred)
    f1=f1_score(y_testlabel, y_pred)
    score_list=[]
    score_list.append(f1)
    score_list.append(g_mean)
    score_list.append(acc)
    score_list.append(auc)
    return score_list

def smoboost_GetScore(pos_train,pos_test,neg_train,neg_test):
    x_train1=np.append(pos_train,neg_train,axis=0)
    y_trainlabel1=[1]*len(pos_train)+[-1]*len(neg_train)
    sm=SMOTE(random_state=0)
    x_train,y_trainlabel=sm.fit_resample(x_train1,y_trainlabel1)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    clf=AdaBoostClassifier(n_estimators=10, random_state=0,algorithm='SAMME')
    
    clf.fit(x_train,y_trainlabel)
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)#这个是带概率的结果，因为AUC的计算要使用
    auc=roc_auc_score(y_testlabel, y_pred_proba[:,1])
    conmat=confusion_matrix(y_testlabel, y_pred)
    tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
    tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
    g_mean=(tpr*tnr)**0.5
    acc=accuracy_score(y_testlabel, y_pred)
    f1=f1_score(y_testlabel, y_pred)
    score_list=[]
    score_list.append(f1)
    score_list.append(g_mean)
    score_list.append(acc)
    score_list.append(auc)
    return score_list

def mmboost_GetScore(pos_train,pos_test,neg_train,neg_test,k,n2pidx):
    x_train=np.append(pos_train,neg_train,axis=0)
    y_trainlabel=[1]*len(pos_train)+[-1]*len(neg_train)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    #clf=Stumpadaboost()
    #clf.fit(x_train,y_trainlabel,k,t=n2pidx)
    clf=AdaBoostClassifier(n_estimators=40, random_state=0)
    clf.fit(x_train,y_trainlabel)
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)#这个是带概率的结果，因为AUC的计算要使用
    auc=roc_auc_score(y_testlabel, y_pred_proba[:,1])
    conmat=confusion_matrix(y_testlabel, y_pred)
    tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
    tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
    g_mean=(tpr*tnr)**0.5
    acc=accuracy_score(y_testlabel, y_pred)
    f1=f1_score(y_testlabel, y_pred)
    score_list=[]
    score_list.append(f1)
    score_list.append(g_mean)
    score_list.append(acc)
    score_list.append(auc)
    return score_list
