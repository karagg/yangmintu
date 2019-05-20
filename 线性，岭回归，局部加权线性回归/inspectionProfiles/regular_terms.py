import numpy as np
from k_fold import k_fold1
from k_fold import random_data
import pandas as pd
def reg_choose(X,y,numval=10):
    """输入特征矩阵和numval交叉验证的次数，返回不同惩罚值的代价函数列表"""
    reg=[0.0,0.01,0.02,0.04,0.1,0.2,0.4,0.8,1.6,3.2,6.4]#列出多个惩罚项值进行比较
    J_train = []
    J_test = []
    for g in reg:
        J_train1, J_test1=k_fold1(X, y, numval=10, lamba=g)#对每个惩罚值进行k折交叉验证
        J_train.append(J_train1)  # 每个degree得到不同的训练集代价函数
        J_test.append(J_test1)  # 每个degree得到不同的验证集代价函数
    df = pd.DataFrame([J_train, J_test], columns=reg, index=['J_train', 'J_test'])#输出不同惩罚值的代价函数列表
    df.columns.names = ["reg"]
    print(df)