import numpy as np
import pandas as pd

def sigmoid(z):
    """ sigmoid函数计算，返回sigmoid函数矩阵"""
    return 1/(1+np.exp(-z))

def computecost(s,y,theta,lamba):
    """计算加入正则化的代价函数，返回一个值J"""
    m=len(y)
    theta1=np.copy(theta)#由于正则化项无theta0,所以对矩阵做出改变
    theta1[0][0]=0
    J=-np.dot(y.T,np.log(s))-np.dot((1-y).T,np.log(1-s))/m+lamba*np.dot(theta1.T,theta)/m/2
    return J

