import numpy as np
def R2(y_true,y_pre):
    """R平方函数，输入y的实际值和预测值，返回R平方"""
    y_true = np.array(y_true)  # 为使-能运行，确保数组为numpy数组
    y_pre = np.array(y_pre)
    e = y_true - y_pre  # 求实际值和预测值之差,残差
    g=y_true-y_true.mean()#求实际值和均值之差，回归差
    R2=1-np.dot(e.T,e)/np.dot(g.T,g)#R平方计算
    return R2