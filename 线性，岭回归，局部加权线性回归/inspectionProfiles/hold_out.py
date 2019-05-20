import numpy as np
import pandas as pd
import gradientdesent as gd
from normalEqu import normalEqu
from MAE import MAE
from MAPE import MAPE
import MSE_RMSE
from R2 import R2
from computeCost import computeCost
from comCost_lwlr import comCost_lwlr
import regression_lwlr as lw
#记得主函数分测试集
def random_data(X,y):
    """此函数为随机重排函数，输入特征变量和标签数组，返回随机重排后的特征变量和标签数组"""
    z=np.hstack([X,y])
    m,n=z.shape
    indexlist=list(range(m))
    np.random.shuffle(indexlist)
    v=z[indexlist,:]
    X=v[:,0:-1]
    y=v[:,-1]
    y=y.reshape(-1,1)
    return X,y
def hold_out(X,y,percent):#注意x,y为numpy数组
    m=len(y)
    train_X=[]
    train_y=[]
    val_X=[]
    val_y=[]
    X,y=random_data(X,y)
    for j in range(m):
        if j<(m*percent):
            train_X.append(X[j])
            train_y.append(y[j])
        else:
            val_X.append(X[j])
            val_y.append(y[j])

    return train_X,train_y,val_X,val_y

def hold_out2(X,y,percent,num_val):
    """留出集评估正规方程函数，输入X特征矩阵，y标签数组，percent训练集所占百分比，num_val几轮验证，输出theta,评估矩阵，返回theta"""
    m=len(y)
    X1=X
    y1=y
    J1=[]#装每轮的训练集代价函数
    J2=[]#装每轮的测试集代价函数
    J5=[[0],[0]]#装每轮的theta
    mae=0
    mape=0
    mse=0
    rmse=0
    r2=0
    for i in range(num_val):
        X1,y1=random_data(X,y)
        q=int(m*percent)
        train_X=X1[:q,:]#按照百分比进行训练集和测试集的切割
        train_y=y1[:q,:]
        val_X=X1[q:,:]
        val_y=y1[q:,:]
        theta,J_train=normalEqu(train_X,train_y)#调用正规方程函数得到代价函数的theta

        J_val=computeCost(val_X,val_y,theta)#得到验证集的代价J
        mae += MAE(val_y, np.dot(val_X,theta))  # 调用MAE函数，进行加和
        mape+= MAPE(val_y, np.dot(val_X,theta))  # 调用MAPE函数
        r2+= R2(val_y, np.dot(val_X,theta))  # 调用R2函数
        mse+= MSE_RMSE.MSE(val_y, np.dot(val_X,theta))  # 调用MSE函数
        rmse += MSE_RMSE.RMSE(val_y, np.dot(val_X,theta))  # 调用RMSE函数
        J1.append(J_train)
        J2.append(J_val)
        J5=np.hstack([J5,theta])
    l,theta=np.hsplit(J5,[1])
    theta=np.mean(theta,axis=1)#几轮下来得到theta平均值
    theta=theta.reshape(2,1)
    print("theta")
    print(theta)#输出theta
    J3=np.mean(J1)#几轮下来得到J_train平均值
    J4=np.mean(J2)#几轮下来得到J_test平均值

    dr = pd.Series([J3,J4,mae/num_val, mape/num_val, mse/num_val, rmse/num_val, r2/num_val], index=["J_train","J_val","MAE", "MAPE", "MSE", "RMSE", "R2"])  # 创立含有七种评估的矩阵
    print(dr)
    return theta


def hold_out3(X,y,percent,num_val,k):
    """留出集评估局部加权线性回归，输入X特征矩阵，y标签数组，percent训练集所占百分比，num_val几轮验证，输出theta,评估矩阵，返回theta"""
    m=len(y)
    X1=X
    y1=y
    J1=[]#装每轮的训练集代价函数
    J2=[]#装每轮的测试集代价函数

    mae=0
    mape=0
    mse=0
    rmse=0
    r2=0
    for i in range(num_val):
        X1,y1=random_data(X,y)
        q=int(m*percent)
        train_X=X1[:q,:]#按照百分比进行训练集和测试集的切割
        train_y=y1[:q,:]
        val_X=X1[q:,:]
        val_y=y1[q:,:]
        y_pre1 = lw.lwlrTest(train_X, train_X, train_y, k)#得到训练集预测值
        y_pre2 = lw.lwlrTest(val_X,train_X, train_y, k)#得到验证集预测值
        J_train=comCost_lwlr(train_y,y_pre1)#得到训练集代价J
        J_val=comCost_lwlr(val_y,y_pre2)#得到验证集的代价J
        mae += MAE(val_y, y_pre2)  # 调用MAE函数，进行加和
        mape+= MAPE(val_y, y_pre2)  # 调用MAPE函数
        r2+= R2(val_y, y_pre2)  # 调用R2函数
        mse+= MSE_RMSE.MSE(val_y, y_pre2)  # 调用MSE函数
        rmse += MSE_RMSE.RMSE(val_y, y_pre2)  # 调用RMSE函数
        J1.append(J_train)
        J2.append(J_val)

    J3=np.mean(J1)#几轮下来得到J_train平均值
    J4=np.mean(J2)#几轮下来得到J_test平均值

    dr = pd.Series([J3,J4,mae/num_val, mape/num_val, mse/num_val, rmse/num_val, r2/num_val], index=["J_train","J_val","MAE", "MAPE", "MSE", "RMSE", "R2"])  # 创立含有七种评估的矩阵
    print(dr)
