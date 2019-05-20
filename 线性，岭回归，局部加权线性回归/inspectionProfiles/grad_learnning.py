import numpy as np
import pandas as pd
from computeCost import computeCost
def grad_learnning(X,y,theta):#此函数编写意义是找到一个最佳学习效率，防止因其过大而跳过代价函数最低点，过小而迭代次数过大须消耗太长时间
    """X为特征矩阵，y为标签数组，theta为角度，此函数功能是粗略找到较合适的学习效率，返回学习效率最佳值"""
    m=len(y)#样本总量
    alpha=[0.001,0.003,0.01,0.03,0.1,0.3,1.0]#规定几个学习效率值，每个大概相差三倍
    num_iters=1500#迭代次数，尽量取大
    a1=[]
    a2=[]
    for i in alpha:
        theta1=theta
        for iters in range(num_iters):
            cha=np.dot(X,theta1)-y#求h(x)-y,得到一个数组
            theta1=theta1-i*(1/m)*np.dot(X.T,cha)#梯度算法应用
        J=computeCost(X,y,theta1)
        a1.append(i)#储存学习效率
        a2.append(J)#储存低于特定值的迭代次数
    df=pd.DataFrame(a2,index=a1,columns=['J'])
    df.index.names=['alpha']

    return df



