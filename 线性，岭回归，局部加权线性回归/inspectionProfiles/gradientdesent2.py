import numpy as np
from computeCost import computeCost

def grad(X,y,alpha,num_iters):
    """X为特征矩阵，y为标签数组，alpha为学习效率，num_iters为所迭代的次数,
    此函数为梯度算法，返回最小角度和代价函数矩阵"""
    m,n=X.shape#样本总量

    theta=np.zeros((n,1))

    for iters in range(num_iters):
        cha=np.dot(X,theta)-y#求h(x)-y,得到一个数组
        theta=theta-alpha*(1/m)*np.dot(X.T,cha)#梯度算法应用
    J=computeCost(X,y,theta)
    return theta,J#返回最佳角度和代价函数矩阵

def Degree(degree,X1):
    X1 = np.array(X1)
    out = np.ones(X1.shape)
    for i in range(1, degree + 1):
        out = np.hstack([out, np.power(X1, i)])
        print(out)
    return out