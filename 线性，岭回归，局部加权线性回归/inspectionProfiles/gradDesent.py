import numpy as np
import pandas as pd
from comCost import computeCost

def grad(X,y,alpha,num_iters,lamba):
    """X为特征矩阵，y为标签数组，theta为角度，alpha为学习效率，num_iters为所迭代的次数,lamba为正则化项
        此函数为梯度算法，返回最小角度和代价函数矩阵"""
    m, n = X.shape  # 样本总量
    theta = np.zeros((n, 1))
    m = len(y)  # 样本总量
    J_history = np.zeros((num_iters, 1))  # 将代价函数矩阵初始为零矩阵
    for iters in range(num_iters):
        cha = np.dot(X, theta) - y  # 求h(x)-y,得到一个数组
        theta1=np.copy(theta)#由于正则化项中不包含theta0，则矩阵要有改变
        theta1[0][0]=0
        theta = theta - alpha * (1 / m) *( np.dot(X.T,cha)-lamba*theta1)# 梯度算法应用
        J_history[iters][0] = computeCost(X, y, theta,lamba)  # 调用代价函数，每次迭代的结果写入，更新矩阵值
    return theta, J_history  # 返回最佳角度和代价函数矩阵
