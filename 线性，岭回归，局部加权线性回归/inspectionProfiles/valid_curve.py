import numpy as np
import pandas as pd
import gradientdesent as gd
import k_fold as kf

def Degree(degree, X1):
    """次方函数，输入最高次幂degree及特征矩阵X1，此矩阵无X0，输出多项式投影后的特征矩阵out"""
    X1=np.array(X1)
    out = np.ones(X1.shape)
    for i in range(1, degree + 1):
        out = np.hstack([out, np.power(X1, i)])
    return out
def Degree2(degree, X1):
    """开方函数，输入最高要开的次方degree及特征矩阵X1，此矩阵无X0，输出多项式投影后的特征矩阵out"""
    X1=np.array(X1)
    out = np.ones(X1.shape)
    X2=X1
    for i in range(1, degree + 1):
        out = np.hstack([out, X2])
        X2=np.sqrt(X2)
    return out
def linear(X,y,degree,numval):
    """回归交叉验证主函数"""
    J_train=[]
    J_test=[]
    for w in degree:
        X1=Degree2(w,X)#通过迭代，不同次幂，不同特征矩阵
        """X1=Degree(w,X)则为次方的"""
        J_train1,J_test1=kf.k_fold2(X1, y, numval)#调用k折交叉验证函数
        J_train.append(J_train1)#每个degree得到不同的训练集代价函数
        J_test.append(J_test1)#每个degree得到不同的验证集代价函数
    df=pd.DataFrame([J_train,J_test],columns=degree,index=['J_train','J_test'])
    df.columns.names=["degress"]
    print(df)#输出显示代价函数与degree的关系矩阵
















X=[[2],[3],[4],[5],[6]]
y=[[1],[2],[3],[4],[5]]
out=Degree(3, X)
#print(out)



