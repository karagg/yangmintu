import numpy as np
from computeCost import computeCost

def normalEqu(X,y):
    """此函数通过正规方程求得当代价函数最小时的最小角，返回由最小theta排列组成的矩阵，其中X是特征矩阵，y是标签数组"""
    X=np.array(X)
    y=np.array(y)
    y.reshape(-1,1)
    turn=np.linalg.pinv(np.dot(X.T,X))#求XT*X的逆
    theta=np.dot(np.dot(turn,X.T),y)#求正规方程函数公式，也就是让代价函数的导数等于零(三维图中达到最凹点）时，theta的计算公式
    J=computeCost(X,y,theta)
    return theta,J#返回角度最佳矩阵


