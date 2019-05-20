import numpy as np
import pandas as pd
def mapFeature(X1,X2,degree):
    """特征映射函数，即输入X1,X2,输出X1，X2，X1^2,X1*X2,X2^2的矩阵,个数多少看输入的degree，注意X1，X2大小相同"""
    X1=X1.reshape(-1,1)
    X2=X2.reshape(-1,1)
    out=np.ones(X1.shape)
    for i in range(1,degree+1):
        for j in range(0,i+1):
            out=np.hstack([out,np.power(X1,i-j)*np.power(X2,j)])#增加行数

    return out



