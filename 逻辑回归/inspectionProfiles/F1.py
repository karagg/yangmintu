import numpy as np
import Confusion_Matrix as cm
def precision(y_true,y_pre):
    """查准率函数，也就是求预测为正例的数据集中，实际也为正例的占比，输入真实值和预测值，输出查准率"""
    tp=cm.TP(y_true,y_pre)#调用TP函数求真正例个数
    fp=cm.FP(y_true,y_pre)#调用FP函数求假正例个数
    P=tp/tp+fp#计算查准率
    return P
def recall(y_true,y_pre):
    """查全率函数，也就是求真实值为正例的数据集中，预测也为正例的占比，输入真实值和预测值，输出查全率"""
    tp = cm.TP(y_true, y_pre)  # 调用TP函数求真正例个数
    fn = cm.FN(y_true, y_pre)  # 调用FN函数求假反例个数
    R= tp / tp + fn  # 计算查全率
    return R
def F1(y_true,y_pre):
    """基于查准率和查全率的调和平均函数，调和平均比较重视较小值，输入真实值和预测值，输出F1值"""
    P=precision(y_true,y_pre)#调用查准率函数
    R=recall(y_true,y_pre)#调用查全率函数
    F1=2*P*R/(P+R)#F1计算
    return F1