import numpy as np
import pandas as pd
from computecost import computecost
from computecost import sigmoid
def gradientDesent(X,y,alpha,num_iter):
    """X为特征矩阵，y为标签数组，theta为角度，alpha为学习效率，num_iters为所迭代的次数,
    此函数为梯度算法，返回最小角度和代价函数矩阵"""
    m, n = X.shape  # 样本总量
    theta = np.zeros((n, 1))
    m=len(y)#样本总量
    theta=theta.reshape(-1,1)
    J_history=np.zeros((num_iter,1))#将代价函数矩阵初始为零矩阵
    for iter in range(num_iter):
        s=sigmoid(np.dot(X,theta))#调用sigmoid函数
        theta=theta-alpha*np.dot(X.T,(s-y))/m#梯度函数应用
        J_history[iter][0]=computecost(s,y)#调用代价函数，每次迭代的结果写入，更新矩阵值
    return theta,J_history
