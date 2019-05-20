import numpy as np
def computeCost(X,y,theta):
    """X为特征矩阵，y为标签数组，theta为角度，此函数为计算代价函数值，返回其值J"""
    m=len(y)#样本总量
    cha=np.dot(X,theta)-y#h(x)-y
    J=np.dot(cha.T,cha)/m/2#代价函数计算
    return J[0]