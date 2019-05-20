import numpy as np
def Accuracy(y_pre, y_true):
    """准确率函数，即预测分类结果和真实值一样的数据量占总数据量的百分比，
    pre为预测值，true为真实值，返回准确率"""
    return sum(y1 == y2 for y1, y2 in zip(y_pre, y_true)) / len(y_true)#准确率计算
