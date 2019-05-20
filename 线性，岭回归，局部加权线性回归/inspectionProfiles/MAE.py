import numpy as np
def MAE(y_true,y_pre):
    """平均绝对误差函数，输入y的实际值和预测值，得到平均绝对误差MAE"""
    y_true=np.array(y_true)#为使-能运行，确保数组为numpy数组
    y_pre=np.array(y_pre)
    e=sum(np.abs(y_true-y_pre))#求其绝对值
    n=len(y_pre)#样本总量
    MAE=e/n#算出MAE
    return MAE