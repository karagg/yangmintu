import numpy as np
import pandas as pd

def sigmoid(z):
    """ sigmoid函数计算，返回sigmoid函数矩阵"""
    return 1/(1+np.exp(-z))

def computecost(s,y):
    """计算代价函数，返回一个J值"""
    m=len(y)
    J=-np.dot(y.T,np.log(s))-np.dot((1-y).T,np.log(1-s))/m
    return J
