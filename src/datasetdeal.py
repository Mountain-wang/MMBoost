import pandas as pd
import numpy as np

def getGlass04vs5():
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-4_vs_5\glass-0-4_vs_5.csv'
    data=pd.read_csv(path)
    positive_data=data[data.label=='positive']
    negative_data=data[data.label!='positive']
    positive_array=positive_data.values
    negative_array=negative_data.values
    positive_array=np.delete(positive_array,9,axis=1)
    negative_array=np.delete(negative_array,9,axis=1)
    return positive_array,negative_array

def getEcoli0346vs5():
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-3-4-6_vs_5\ecoli-0-3-4-6_vs_5.csv'
    data=pd.read_csv(path)
    positive_data=data[data.label=='positive']
    negative_data=data[data.label!='positive']
    positive_array=positive_data.values
    negative_array=negative_data.values
    target_col=len(positive_array[0])
    positive_array=np.delete(positive_array,target_col-1,axis=1)
    negative_array=np.delete(negative_array,target_col-1,axis=1)
    return positive_array,negative_array

def getEcoli0347vs56():
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-3-4-7_vs_5-6\ecoli-0-3-4-7_vs_5-6.csv'
    data=pd.read_csv(path)
    positive_data=data[data.label=='positive']
    negative_data=data[data.label!='positive']
    positive_array=positive_data.values
    negative_array=negative_data.values
    target_col=len(positive_array[0])
    positive_array=np.delete(positive_array,target_col-1,axis=1)
    negative_array=np.delete(negative_array,target_col-1,axis=1)
    return positive_array,negative_array

def getYeast05679vs4():
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-0-5-6-7-9_vs_4\yeast-0-5-6-7-9_vs_4.csv'
    data=pd.read_csv(path)
    positive_data=data[data.label=='positive']
    negative_data=data[data.label!='positive']
    positive_array=positive_data.values
    negative_array=negative_data.values
    target_col=len(positive_array[0])
    positive_array=np.delete(positive_array,target_col-1,axis=1)
    negative_array=np.delete(negative_array,target_col-1,axis=1)
    return positive_array,negative_array

def getEcoli067vs5():
    path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-6-7_vs_5\ecoli-0-6-7_vs_5.csv'
    data=pd.read_csv(path)
    positive_data=data[data.label=='positive']
    negative_data=data[data.label!='positive']
    positive_array=positive_data.values
    negative_array=negative_data.values
    target_col=len(positive_array[0])
    positive_array=np.delete(positive_array,target_col-1,axis=1)
    negative_array=np.delete(negative_array,target_col-1,axis=1)
    return positive_array,negative_array

def getTargetDdata(path):
    data=pd.read_csv(path)
    positive_data=data[data.label=='positive']
    negative_data=data[data.label!='positive']
    positive_array=positive_data.values
    negative_array=negative_data.values
    target_col=len(positive_array[0])
    positive_array=np.delete(positive_array,target_col-1,axis=1)
    negative_array=np.delete(negative_array,target_col-1,axis=1)
    return positive_array,negative_array
    
def get33Data(data_num):
    print('现在是33--',data_num,'数据集上的实验')
    #===========================================================================
    # print('现在进行的是新的33个数据集上的情况')
    # data_name=input('请输入使用的第几个数据集，从0开始，32结束')
    # data_num=int(data_name)
    #===========================================================================
    if data_num==0:
        positive_array,negative_array=getGlass04vs5()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-4_vs_5\result.csv'
        k=1.25   #可能是1.3
    elif data_num==1:
        positive_array,negative_array=getEcoli0346vs5()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-3-4-6_vs_5\result.csv'
        k=1.42
    elif data_num==2:
        positive_array,negative_array=getEcoli0347vs56()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-3-4-7_vs_5-6\result.csv'
        k=1.4
    elif data_num==3:
        positive_array,negative_array=getYeast05679vs4()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-0-5-6-7-9_vs_4\result.csv'
        k=0.9
    elif data_num==4:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-6-7_vs_5\ecoli-0-6-7_vs_5.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-6-7_vs_5\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=1.25
    elif data_num==5:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\vowel\vowel0\vowel0.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\vowel\vowel0\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=1.17
    elif data_num==6:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_2\glass-0-1-6_vs_2.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_2\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==7:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass2\glass2.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass2\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=1.1
    elif data_num==8:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_2-3-5-6\ecoli-0-1-4-7_vs_2-3-5-6.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_2-3-5-6\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=0.92
    elif data_num==9:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\led7digit-0-2-4-5-6-7-8-9_vs_1\led7digit-0-2-4-5-6-7-8-9_vs_1.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\led7digit-0-2-4-5-6-7-8-9_vs_1\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=1.04
    elif data_num==10:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1_vs_5\ecoli-0-1_vs_5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1_vs_5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.4
    elif data_num==11:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-6_vs_5\glass-0-6_vs_5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-6_vs_5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.11
    elif data_num==12:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-4-6_vs_2\glass-0-1-4-6_vs_2.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-4-6_vs_2\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.86
    elif data_num==13:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_5-6\ecoli-0-1-4-7_vs_5-6.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_5-6\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.28
    elif data_num==14:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\cleveland-0_vs_4\cleveland-0_vs_4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\cleveland-0_vs_4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.35
    elif data_num==15:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-6_vs_5\ecoli-0-1-4-6_vs_5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-6_vs_5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.4
    elif data_num==19:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli4\ecoli4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.22
    elif data_num==16:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c0-vs-c4\shuttle-c0-vs-c4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c0-vs-c4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.82
    elif data_num==17:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1_vs_7\yeast-1_vs_7.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1_vs_7\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.1
    elif data_num==18:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass4\glass4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.9
    elif data_num==20:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\page-blocks-1-3_vs_4\page-blocks-1-3_vs_4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\page-blocks-1-3_vs_4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.8
    elif data_num==21:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone9-18\abalone9-18.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone9-18\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.87
    elif data_num==22:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_5\glass-0-1-6_vs_5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.05
    elif data_num==23:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c2-vs-c4\shuttle-c2-vs-c4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c2-vs-c4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==24:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-4-5-8_vs_7\yeast-1-4-5-8_vs_7.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-4-5-8_vs_7\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.01
    elif data_num==25:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass5\glass5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.93
    elif data_num==26:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-2_vs_8\yeast-2_vs_8.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-2_vs_8\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.22
    elif data_num==27:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast4\yeast4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.12
    elif data_num==28:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-2-8-9_vs_7\yeast-1-2-8-9_vs_7.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-2-8-9_vs_7\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.95
    elif data_num==29:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast5\yeast5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.9
    elif data_num==30:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-3-7_vs_2-6\ecoli-0-1-3-7_vs_2-6.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-3-7_vs_2-6\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.01
    elif data_num==31:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast6\yeast6.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast6\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.8
    elif data_num==32:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone19\abalone19.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone19\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        
        #接下来是新加的大数据集
    elif data_num==37:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\covtype\covtype3v4\covtype3v4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\covtype\covtype3v4\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==34:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\page-blocks-1-3_vs_4\page-blocks0\page-blocks0.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\page-blocks-1-3_vs_4\page-blocks0\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==40:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-z\letter-z.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-z\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==33:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\penbased\penbased2\penbased2.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\penbased\penbased2\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==35:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\penbased\penbased5\penbased5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\penbased\penbased5\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==36:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-ch\letter-ch.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-ch\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==38:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone6\abalone6.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone6\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==41:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\covtype\covtype4v3567\covtype4v3567.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\covtype\covtype4v3567\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==42:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone4\abalone4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone4\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    elif data_num==39:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-u\letter-u.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-u\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
    
    else:
        print('misdataset')
        exit()
    
    return positive_array,negative_array,resultpath,k


#===============================================================================
# def onlyoneData():#这个数据集的顺序有问题
#     print('现在进行的是新的33个数据集上的情况')
#     data_name=input('请输入使用的第几个数据集，从0开始，32结束')
#     data_num=int(data_name)
#     if data_num==0:
#         positive_array,negative_array=getGlass04vs5()
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-4_vs_5\result.csv'
#         k=1   #可能是1.3
#     elif data_num==1:
#         positive_array,negative_array=getEcoli0346vs5()
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-3-4-6_vs_5\result.csv'
#         k=1
#     elif data_num==2:
#         positive_array,negative_array=getEcoli0347vs56()
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-3-4-7_vs_5-6\result.csv'
#         k=1
#     elif data_num==3:
#         positive_array,negative_array=getYeast05679vs4()
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-0-5-6-7-9_vs_4\result.csv'
#         k=1
#     elif data_num==4:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-6-7_vs_5\ecoli-0-6-7_vs_5.csv'
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-6-7_vs_5\result.csv'
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==5:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\vowel\vowel0\vowel0.csv'
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\vowel\vowel0\result.csv'
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==6:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_2\glass-0-1-6_vs_2.csv'
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_2\result.csv'
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==7:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass2\glass2.csv'
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass2\result.csv'
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==8:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_2-3-5-6\ecoli-0-1-4-7_vs_2-3-5-6.csv'
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_2-3-5-6\result.csv'
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==9:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\led7digit-0-2-4-5-6-7-8-9_vs_1\led7digit-0-2-4-5-6-7-8-9_vs_1.csv'
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\led7digit-0-2-4-5-6-7-8-9_vs_1\result.csv'
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==10:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1_vs_5\ecoli-0-1_vs_5.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1_vs_5\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==11:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-6_vs_5\glass-0-6_vs_5.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-6_vs_5\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==12:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-4-6_vs_2\glass-0-1-4-6_vs_2.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-4-6_vs_2\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==13:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_5-6\ecoli-0-1-4-7_vs_5-6.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_5-6\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==14:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\cleveland-0_vs_4\cleveland-0_vs_4.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\cleveland-0_vs_4\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==15:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-6_vs_5\ecoli-0-1-4-6_vs_5.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-6_vs_5\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==16:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli4\ecoli4.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli4\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==17:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c0-vs-c4\shuttle-c0-vs-c4.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c0-vs-c4\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==18:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1_vs_7\yeast-1_vs_7.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1_vs_7\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==19:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass4\glass4.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass4\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==20:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\page-blocks-1-3_vs_4\page-blocks-1-3_vs_4.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\page-blocks-1-3_vs_4\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==21:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone9-18\abalone9-18.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone9-18\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==22:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_5\glass-0-1-6_vs_5.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_5\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==23:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c2-vs-c4\shuttle-c2-vs-c4.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c2-vs-c4\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==24:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-4-5-8_vs_7\yeast-1-4-5-8_vs_7.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-4-5-8_vs_7\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==25:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass5\glass5.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass5\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==26:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-2_vs_8\yeast-2_vs_8.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-2_vs_8\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==27:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast4\yeast4.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast4\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==28:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-2-8-9_vs_7\yeast-1-2-8-9_vs_7.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-2-8-9_vs_7\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==29:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast5\yeast5.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast5\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==30:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-3-7_vs_2-6\ecoli-0-1-3-7_vs_2-6.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-3-7_vs_2-6\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==31:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast6\yeast6.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast6\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#     elif data_num==32:
#         path=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone19\abalone19.csv' 
#         resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone19\result.csv' 
#         positive_array,negative_array=getTargetDdata(path)
#         k=1
#         
#     else:
#         print('misdataset')
#         exit()
#     
#     return positive_array,negative_array,resultpath,k
#===============================================================================


def getMoreLargeData(data_num):
    print('现在是33--',data_num,'数据集上的实验')
    #===========================================================================
    # print('现在进行的是新的33个数据集上的情况')
    # data_name=input('请输入使用的第几个数据集，从0开始，32结束')
    # data_num=int(data_name)
    #===========================================================================
    if data_num==0:
        positive_array,negative_array=getGlass04vs5()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-4_vs_5\result.csv'
        k=1.25   #可能是1.3
    elif data_num==1:
        positive_array,negative_array=getEcoli0346vs5()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-3-4-6_vs_5\result.csv'
        k=1.42
    elif data_num==2:
        positive_array,negative_array=getEcoli0347vs56()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-3-4-7_vs_5-6\result.csv'
        k=1.4
    elif data_num==3:
        positive_array,negative_array=getYeast05679vs4()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-0-5-6-7-9_vs_4\result.csv'
        k=0.9
    
    return positive_array,negative_array,resultpath,k




def getRKon33Data(data_num):
    print('现在是33--',data_num,'数据集上的实验')
    #===========================================================================
    # print('现在进行的是新的33个数据集上的情况')
    # data_name=input('请输入使用的第几个数据集，从0开始，32结束')
    # data_num=int(data_name)
    #===========================================================================
    if data_num==0:
        positive_array,negative_array=getGlass04vs5()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-4_vs_5\result.csv'
        k=1.25   #可能是1.3
        R=1
    elif data_num==1:
        positive_array,negative_array=getEcoli0346vs5()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-3-4-6_vs_5\result.csv'
        k=1.42
        R=1
    elif data_num==2:
        positive_array,negative_array=getEcoli0347vs56()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-3-4-7_vs_5-6\result.csv'
        k=1.4
        R=1
    elif data_num==3:
        positive_array,negative_array=getYeast05679vs4()
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-0-5-6-7-9_vs_4\result.csv'
        k=0.9
        R=1
    elif data_num==4:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-6-7_vs_5\ecoli-0-6-7_vs_5.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-6-7_vs_5\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=1.25
        R=1
    elif data_num==5:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\vowel\vowel0\vowel0.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\vowel\vowel0\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=1.17
        R=1
    elif data_num==6:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_2\glass-0-1-6_vs_2.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_2\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==7:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass2\glass2.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass2\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=1.1
        R=1
    elif data_num==8:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_2-3-5-6\ecoli-0-1-4-7_vs_2-3-5-6.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_2-3-5-6\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=0.92
        R=1
    elif data_num==9:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\led7digit-0-2-4-5-6-7-8-9_vs_1\led7digit-0-2-4-5-6-7-8-9_vs_1.csv'
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\led7digit-0-2-4-5-6-7-8-9_vs_1\result.csv'
        positive_array,negative_array=getTargetDdata(path)
        k=1.04
        R=1
    elif data_num==10:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1_vs_5\ecoli-0-1_vs_5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1_vs_5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.4
        R=1
    elif data_num==11:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-6_vs_5\glass-0-6_vs_5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-6_vs_5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.11
        R=1
    elif data_num==12:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-4-6_vs_2\glass-0-1-4-6_vs_2.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-4-6_vs_2\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.86
        R=1
    elif data_num==13:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_5-6\ecoli-0-1-4-7_vs_5-6.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-7_vs_5-6\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.28
        R=1
    elif data_num==14:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\cleveland-0_vs_4\cleveland-0_vs_4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\cleveland-0_vs_4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.35
        R=1
    elif data_num==15:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-6_vs_5\ecoli-0-1-4-6_vs_5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-4-6_vs_5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.4
        R=1
    elif data_num==19:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli4\ecoli4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.22
        R=1
    elif data_num==16:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c0-vs-c4\shuttle-c0-vs-c4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c0-vs-c4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.82
        R=1
    elif data_num==17:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1_vs_7\yeast-1_vs_7.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1_vs_7\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.1
        R=1
    elif data_num==18:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass4\glass4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.9
        R=1
    elif data_num==20:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\page-blocks-1-3_vs_4\page-blocks-1-3_vs_4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\page-blocks-1-3_vs_4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.8
        R=1
    elif data_num==21:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone9-18\abalone9-18.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone9-18\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.87
        R=2
    elif data_num==22:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_5\glass-0-1-6_vs_5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass-0-1-6_vs_5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.05
        R=1
    elif data_num==23:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c2-vs-c4\shuttle-c2-vs-c4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\shuttle\shuttle-c2-vs-c4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==24:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-4-5-8_vs_7\yeast-1-4-5-8_vs_7.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-4-5-8_vs_7\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.01
        R=4
    elif data_num==25:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass5\glass5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\glass\glass5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.93
        R=1
    elif data_num==26:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-2_vs_8\yeast-2_vs_8.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-2_vs_8\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.22
        R=1
    elif data_num==27:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast4\yeast4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast4\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.12
        R=3
    elif data_num==28:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-2-8-9_vs_7\yeast-1-2-8-9_vs_7.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast-1-2-8-9_vs_7\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.95
        R=1
    elif data_num==29:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast5\yeast5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast5\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.9
        R=1
    elif data_num==30:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-3-7_vs_2-6\ecoli-0-1-3-7_vs_2-6.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\ecoli\ecoli-0-1-3-7_vs_2-6\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1.01
        R=1
    elif data_num==31:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast6\yeast6.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\yeast\yeast6\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=0.8
        R=1
    elif data_num==32:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone19\abalone19.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone19\result.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
        
    ######后面就是大数据集了点
    elif data_num==37:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\covtype\covtype3v4\covtype3v4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\covtype\covtype3v4\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==34:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\page-blocks-1-3_vs_4\page-blocks0\page-blocks0.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\page-blocks-1-3_vs_4\page-blocks0\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==40:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-z\letter-z.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-z\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==33:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\penbased\penbased2\penbased2.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\penbased\penbased2\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==35:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\penbased\penbased5\penbased5.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\penbased\penbased5\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==36:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-ch\letter-ch.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-ch\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==38:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone6\abalone6.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone6\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==41:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\covtype\covtype4v3567\covtype4v3567.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\covtype\covtype4v3567\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==42:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone4\abalone4.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\abalone\abalone4\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
    elif data_num==39:
        path=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-u\letter-u.csv' 
        resultpath=r'D:\Codetool\datasrc\imbalancedata\moredata\letter\letter-u\results.csv' 
        positive_array,negative_array=getTargetDdata(path)
        k=1
        R=1
        
    else:
        print('misdataset')
        exit()
    
    return positive_array,negative_array,resultpath,k,R


if __name__=='__main__':
    #positive_array,negative_array=getGlass04vs5()
    while True:
        dataset_num=input('请输入使用的第几个数据集，从0--32选择')
        if dataset_num=='-1':
            exit(0)
        positive_array,negative_array,_,_ =get33Data(int(dataset_num))
        print(positive_array[0,:])        
        print(negative_array[0,:])
        print(len(positive_array[0]))
        print(len(positive_array))
        print(len(negative_array))
        print(len(positive_array)+len(negative_array))