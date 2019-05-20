import numpy as np
from computeCost import computeCost
def gradientDesent(X,y,theta,alpha,num_iters):
    """X为特征矩阵，y为标签数组，theta为角度alpha为学习效率，num_iters为所迭代的次数,
    此函数为梯度算法，返回最小角度和代价函数矩阵"""
    m=len(y)#样本总量
    J_history=np.zeros((num_iters,1))#将代价函数矩阵初始为零矩阵
    for iters in range(num_iters):
        cha=np.dot(X,theta)-y#求h(x)-y,得到一个数组
        theta=theta-alpha*(1/m)*np.dot(X.T,cha)#梯度算法应用
        J_history[iters][0]=computeCost(X,y,theta)#调用代价函数，每次迭代的结果写入，更新矩阵值
    return theta,J_history#返回最佳角度和代价函数矩阵



