import numpy as np
import pandas as pd
from computeCost import computeCost
def O_u_fit(X_train,y_train,X_val,y_val,theta):
    """过拟合和欠拟合评估"""
    train_J=computeCost(X_train,y_train,theta)#计算训练集代价函数
    val_J=computeCost(X_val,y_val,theta)#计算测试集代价函数
    dg=pd.Series([train_J,val_J],index=["train_J","val_J"])#合并显示
    print(dg)
    print("一般情况下，如果train_J<<val_J,则是过拟合，如果train_J约等于val_J,且其值都比较大，则为欠拟合")
    
    