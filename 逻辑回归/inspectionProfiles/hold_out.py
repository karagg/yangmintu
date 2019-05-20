import numpy as np
import pandas as pd


from gradientDesent import gradientDesent
from computecost import computecost
from computecost import sigmoid
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


def hold_out(X,y,percent,num_val=10):
    """留出集评估逻辑回归函数，输入X特征矩阵，y标签数组，percent训练集所占百分比，num_val几轮验证，输出theta,评估矩阵，返回theta"""
    m=len(y)
    X1=X
    y1=y
    J1=[]#装每轮的训练集代价函数
    J2=[]#装每轮的测试集代价函数
    J5=[[0],[0],[0]]#装每轮的theta

    for i in range(num_val):
        X1,y1=random_data(X,y)
        q=int(m*percent)
        train_X=X1[:q,:]#按照百分比进行训练集和测试集的切割
        train_y=y1[:q,:]
        val_X=X1[q:,:]
        val_y=y1[q:,:]
        theta,J_train=gradientDesent(train_X,train_y,0.3,1500)#调用梯度下降逻辑回归函数得到代价函数的theta
        s = sigmoid(np.dot(val_X, theta))
        J_val=computecost(s,val_y)#得到验证集的代价J

        J1.append(J_train[-1,-1])
        J2.append(J_val)
        J5=np.hstack([J5,theta])
    l,theta=np.hsplit(J5,[1])
    theta=np.mean(theta,axis=1)#几轮下来得到theta平均值
    theta=theta.reshape(3,1)
    print("theta")
    print(theta)#输出theta
    J3=np.mean(J1)#几轮下来得到J_train平均值
    J4=np.mean(J2)#几轮下来得到J_test平均值
    df=pd.Series([J3,J4],index=['J_train',"J_test"])
    print(df)
    return theta