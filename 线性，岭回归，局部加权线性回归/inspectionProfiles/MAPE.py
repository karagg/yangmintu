import numpy as np
def MAPE(y_true,y_pre):
    """平均绝对百分误差函数，输入y的实际值和预测值，得到平均绝对百分误差MAPE"""
    y_true=np.array(y_true)#为使-能运行，确保数组为numpy数组
    y_pre=np.array(y_pre)
    e=np.abs(y_true-y_pre)#求其绝对值
    n=len(y_pre)#样本总量
    p=np.abs(y_true)
    f=sum(e/p)
    MAPE=f/n#平均绝对百分误差计算
    return MAPE