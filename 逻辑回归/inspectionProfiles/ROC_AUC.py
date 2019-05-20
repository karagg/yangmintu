import numpy as np
import Confusion_Matrix as cm
import matplotlib.pyplot as plt
def ROC(y_true,y_pre_s,thresholds):
    """受试者工作特征函数，是基于假正例率与真正例率的数组，输入y_true真实值，y_pre_s预测的正例概率，thresholds为分类阈值（可是一个值或一维数组），
    输出为由假正例率与真正例率组成的数组"""
    ret=[[0,0]]
    for threshold in thresholds:
        y_pre = y_pre_s.copy()
        y_pre[y_pre_s<threshold]=0#根据阈值确定预测值
        y_pre[y_pre_s>=threshold]=1
        tp=cm.TP(y_true,y_pre)#调用TP函数
        fn=cm.FN(y_true,y_pre)#调用FN函数
        fp=cm.FP(y_true,y_pre)#调用FP函数
        tn=cm.TN(y_true,y_pre)#调用TN函数
        TPR=tp/(tp+fn)#求得真正例率
        FPR=fp/(tn+fp)#求得假正例率
        ret.append([FPR,TPR])
    ret=np.array(ret)
    plt.plot(np.sort(ret[:,0]),np.sort(ret[:,1]),'-')#显示ROC曲线，x轴为假正例率，y轴为真正例率
    plt.title("ROC")
    plt.show()

    return ret
def AUC(y_true,y_pre_s,thresholds):
    """即ROC曲线下的面积函数，输入y_true真实值，y_pre_s预测的正例概率，thresholds为分类阈值（可是一个值或一维数组），输出面积值"""
    ret=ROC(y_true,y_pre_s,thresholds)#调用ROC函数
    fpr=np.sort(ret[:,0])#对真正例率和假正例率进行重排序
    tpr=np.sort(ret[:,1])
    m=len(thresholds)#阈值个数
    AUC=0
    for i in range(m-1):
        AUC+=(tpr[i+1]+tpr[i])*(fpr[i+1]-fpr[i])/2#计算每小块面积加和
    print("AUC")
    return AUC



