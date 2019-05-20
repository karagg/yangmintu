import numpy as np
def MSE(y_true,y_pre):
    """均方误差函数，输入y的实际值和预测值，返回均方误差MSE"""
    y_true=np.array(y_true)#为使-能运行，确保数组为numpy数组
    y_pre=np.array(y_pre)
    e=y_true-y_pre#求实际值和预测值之差
    n=len(y_pre)#样本总量
    MSE=np.dot(e.T,e)/n#计算均方误差
    return MSE
def RMSE(y_true,y_pre):
    """均方根误差函数，输入y的实际值和预测值，返回均方根误差MSE"""
    e=MSE(y_true,y_pre)#调用均方误差函数
    RMSE=np.sqrt(e)#均方根误差计算
    return RMSE