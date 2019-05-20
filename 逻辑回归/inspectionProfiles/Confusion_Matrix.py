import numpy as np
import pandas as pd
def TP(y_true,y_pre):
    """为求真正例函数，也就是预测值和实际值都为正例的函数，输入y_true真实值，y_pre预测值，返回真正例个数TP"""
    true_positive = sum(y1 and int(y2) for y1, y2 in zip(y_true,y_pre))
    return true_positive[0]

def TN(y_true,y_pre):
    """为求真反例例函数，也就是预测值和实际值都为反例的函数，输入y_true真实值，y_pre预测值，返回真反例个数TN"""
    true_negative=sum(1-(y1 or int(y2) )for y1, y2 in zip(y_true,y_pre))
    return true_negative[0]
def FN(y_true,y_pre):
    """为求假反例例函数，也就是预测值为反，实际值为正的函数，输入y_true真实值，y_pre预测值，返回假反例个数FN"""
    tp=TP(y_true,y_pre)#调用函数求得真正例个数
    n=np.sum(y_true==1)#操作布尔数组求得真实值为正的个数
    false_negative=n-tp#真实值为正减去真正例等于假反例个数
    return false_negative
def FP(y_true,y_pre):
    """为求假正例例函数，也就是预测值为正，实际值为反的函数，输入y_true真实值，y_pre预测值，返回假正例个数FN"""
    tn=TN(y_true,y_pre)#调用函数求得真反例个数
    n = np.sum(y_true == 0)  # 操作布尔数组求得真实值为反的个数
    false_positive=n-tn#真实值为假减去真反例等于假正例个数
    return false_positive
def Con_Mat(y_true,y_pre):
    """显示混淆矩阵函数，输入y_true真实值，y_pre预测值，输出混淆矩阵数组"""
    tp = TP(y_true, y_pre)  # 调用函数求得真正例个数
    tn = TN(y_true, y_pre)  # 调用函数求得真反例个数
    fp=FP(y_true, y_pre)#调用函数求得假正例个数
    fn=FN(y_true, y_pre)#调用函数求得假反例个数
    df=pd.DataFrame([[tp,fn],[fp,tn]],index=['true','false'],columns=['true','false'])#创建混淆矩阵
    df.index.names=['true_sit']
    df.columns.names=['pre_sit']
    print(df)#输出混淆矩阵