import numpy as np
def comCost_lwlr(y_test,y_pre):
    m = len(y_test)  # 样本总量
    cha = y_pre-y_test # h(x)-y
    J = np.dot(cha.T, cha) / m / 2  # 代价函数计算
    return J