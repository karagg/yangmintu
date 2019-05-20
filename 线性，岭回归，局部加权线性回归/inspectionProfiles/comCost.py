import numpy as np
import pandas as pd
def computeCost(X,y,theta,lamba):
    """X为特征矩阵，y为标签数组，theta为角度，lamba为正则化项，此函数为计算代价函数值，返回其值J"""
    m = len(y)  # 样本总量
    theta=theta.reshape(-1,1)#使其变成列向量，用于后面矩阵运算
    theta1=np.copy(theta)#由于正则化项中不包含theta0，则矩阵要有改变
    theta1[0][0]=0
    cha = np.dot(X, theta) - y  # h(x)-y
    J = (np.dot(cha.T , cha)+lamba*np.dot(theta1.T,theta1))/ m / 2  # 加入正则化项代价函数计算
    return J
