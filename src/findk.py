from modifytest import neg2posInOneTime
from myadaboost import *
from datasetdeal import get33Data
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import f1_score,confusion_matrix,roc_auc_score,accuracy_score

def onetraintoFindK():
    dataset_num=input('请输入使用的第几个数据集，从0--32选择')
    posarray,negarray,resultpath,k=get33Data(int(dataset_num))

    clfmodel='bs'
    kstar=float(input('请输入k的初始值'))
    kfoot=float(input('请输入k的步长'))
    kend=float(input('请输入k的终止值'))
    score_list=[]
    pos_train,pos_test=train_test_split(posarray,test_size=0.2)#0.1
    neg_train,neg_test=train_test_split(negarray,test_size=0.2)
    #easyeb(pos_train, pos_test, neg_train, neg_test)
    k=kstar
    while k<kend+0.5*kfoot:#这样操作是为了避免计算机是整数，可能出现1.50000001的情况，
        eachscore_list=lookup_n_InOneTime(pos_train, pos_test, neg_train, neg_test, clfmodel, k)
        k=k+kfoot
        score_list.append(eachscore_list)
    
    for i in range(len(score_list[0])):
        #print('********')
        print('********',i,i,i,i,i,i,'倍时的结果*******')
        print('先打印f-measure')
        for j in range(len(score_list)):
            print('k=',round(kstar+j*kfoot,2),': ',score_list[j][i][0])
            #print(i,'倍 ','k=',kstar+j*kfoot,'时f1值：',score_list[j][i][0])
            #print(i,'倍 ','k=',kstar+j*kfoot,'时gmean值：',score_list[j][i][1])
            #print(i,'倍 ','k=',kstar+j*kfoot,'时acc值：',score_list[j][i][2])
        print('再打印g-measure')
        for j in range(len(score_list)):
            
            print('k=',round(kstar+j*kfoot,2),': ',score_list[j][i][1])
        
        print('再打印精确度acc')
        for j in range(len(score_list)):
            
            print('k=',round(kstar+j*kfoot,2),': ',score_list[j][i][2])
            
        print('再打印模型比较AUC')
        for j in range(len(score_list)):
            
            print('k=',round(kstar+j*kfoot,2),': ',score_list[j][i][3])
    
  
    pass  

def lookup_n_InOneTime(pos_train,pos_test,neg_train,neg_test,clfmodel,k=1):
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    clf_list=[]
    n2p_index=0
    posarray_list,negarray_list=neg2posInOneTime(pos_train, neg_train)
    for eachpos_train,eachneg_train in zip(posarray_list,negarray_list):
        each_train=np.append(eachpos_train,eachneg_train,axis=0)
        each_label=[1]*len(eachpos_train)+[-1]*len(eachneg_train)
        eachclf=Stumpadaboost()
            
            #n2p_index=n2p_index+len(pos_train)
        eachclf.fit(each_train,each_label,k,t=n2p_index)
        n2p_index=n2p_index+len(pos_train)
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
        acc=accuracy_score(y_testlabel, eachclf_predict)
        aucscore=roc_auc_score(y_testlabel,eachclf_predict_proba[:,1])
        print('n=',j,'情况下混淆矩阵:','\n',conmat)
        print('n=',j,'情况下f1:',f1score)
        print('n=',j,'情况下G-MEAN:',g_mean)
        print('n=',j,'情况下整体精确度:',acc)
        print('n=',j,'情况下模型AUC:',aucscore)
        score_list=[]
        score_list.append(f1score)
        score_list.append(g_mean)
        score_list.append(accuracy_score(y_testlabel, eachclf_predict))
        score_list.append(aucscore)
        allscore_list.append(score_list)

    
    return allscore_list 



def oneFindKWithN(pos_train,pos_test,neg_train,neg_test,kstar,kfoot,kend,bestn):
    kscore_list=[]
    #pos_train,pos_test=train_test_split(posarray,test_size=0.2)#0.1
    #neg_train,neg_test=train_test_split(negarray,test_size=0.2)
    x_test=np.append(pos_test,neg_test,axis=0)
    y_testlabel=[1]*len(pos_test)+[-1]*len(neg_test)
    n2p_index=0
    posarray_list,negarray_list=neg2posInOneTime(pos_train, neg_train)
    eachpos_train,eachneg_train=posarray_list[bestn],negarray_list[bestn]
    each_train=np.append(eachpos_train,eachneg_train,axis=0)
    each_label=[1]*len(eachpos_train)+[-1]*len(eachneg_train)
    n2p_index=n2p_index+len(pos_train)
    

    k=kstar
    while k<kend+0.5*kfoot:#这样操作是为了避免计算机是整数，可能出现1.50000001的情况，
        eachclf=Stumpadaboost()
        eachclf.fit(each_train,each_label,k,t=n2p_index)
        eachclf_predict=eachclf.predict(x_test)
        eachclf_predict_proba=eachclf.predict_proba(x_test)
        conmat=confusion_matrix(y_testlabel,eachclf_predict)
        tnr=conmat[0][0]/(conmat[0][0]+conmat[0][1])
        tpr=conmat[1][1]/np.sum(conmat,axis=1)[1]
        g_mean=(tpr*tnr)**0.5
        f1score=f1_score(y_testlabel, eachclf_predict)
        acc=accuracy_score(y_testlabel, eachclf_predict)
        aucscore=roc_auc_score(y_testlabel,eachclf_predict_proba[:,1])
        score_list=[]
        score_list.append(f1score)
        score_list.append(g_mean)
        score_list.append(acc)
        score_list.append(aucscore)
        
        k=k+kfoot
        kscore_list.append(score_list)
    
    #===========================================================================
    # print('先打印出f1')
    # for j in range(len(kscore_list)):
    #     print(kscore_list[j][0])
    # print('先打印出gmean')
    # for j in range(len(kscore_list)):
    #     print(kscore_list[j][1])
    # print('先打印出auc')
    # for j in range(len(kscore_list)):
    #     print(kscore_list[j][3])
    #===========================================================================
    
    return kscore_list
            
def FindKWithN():
    dataset_num=input('请输入使用的第几个数据集，从0--32选择')
    posarray,negarray,resultpath,k=get33Data(int(dataset_num))

    clfmodel='bs'
    bestn=1
    kstar=float(input('请输入k的初始值'))
    kfoot=float(input('请输入k的步长'))
    kend=float(input('请输入k的终止值'))
    kf=KFold(5,shuffle=True)
    allkfscore_list=[]
    for (pos_trainindex,pos_testindex),(neg_trainindex,neg_testindex) in zip(kf.split(posarray),kf.split(negarray)):
        pos_train,pos_test=posarray[pos_trainindex],posarray[pos_testindex]
        neg_train,neg_test=negarray[neg_trainindex],negarray[neg_testindex]
        eachscore_list=oneFindKWithN(pos_train,pos_test,neg_train,neg_test,kstar,kfoot,kend,bestn)
        allkfscore_list.append(eachscore_list)
        
    print('先打印出f1d的均值')
    for k in range(len(allkfscore_list[0])):
        sum=0
        for i in range(len(allkfscore_list)):
            sum=sum+allkfscore_list[i][k][0]
        avg=sum/len(allkfscore_list)
        print(avg)
    print('先打印出gmean的均值')
    for k in range(len(allkfscore_list[0])):
        sum=0
        for i in range(len(allkfscore_list)):
            sum=sum+allkfscore_list[i][k][1]
        avg=sum/len(allkfscore_list)
        print(avg)
    print('先打印出auc的均值')
    for k in range(len(allkfscore_list[0])):
        sum=0
        for i in range(len(allkfscore_list)):
            sum=sum+allkfscore_list[i][k][3]
        avg=sum/len(allkfscore_list)
        print(avg)
    


if __name__=='__main__':
    #onetraintoFindK()  
    FindKWithN()