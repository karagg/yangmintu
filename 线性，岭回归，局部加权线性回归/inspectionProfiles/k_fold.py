import numpy as np
import pandas as pd
from computeCost import computeCost
from normalEqu_Reg  import normalEqu_Reg
import  gradientdesent2 as gd
from R2 import R2
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
def k_fold1(X,y,numval=10,lamba=0):
    """岭回归法的k折交叉验证，lamba为惩罚值"""
    y = np.array(y)
    m = len(y)
    g = int(m / numval)
    J3 = []  # 创建装每次训练集代价函数的空列表
    J6 = []  # 创建装每次测试集代价函数的空列表
    for i in range(numval):
        X, y = random_data(X, y)  # k次k折，每一次都会随机打乱顺序重排
        J1 = []  # 创建装一次中每轮训练集代价函数的空列表
        J4 = []  # 创建装一次中每轮测试集代价函数的空列表
        for j in range(numval):
            """分开训练集和验证集，得到各自的代价函数"""
            if j == 0:

                X_test = X[:(j + 1) * g, :]
                y_test = y[:(j + 1) * g, :]
                X_train = X[(j + 1) * g:, :]
                y_train = y[(j + 1) * g:, :]
                theta1= normalEqu_Reg(X_train,y_train,lamba)
                J_train= computeCost(X_train, y_train, theta1)
                J_test = computeCost(X_test, y_test, theta1)
                # J_train = R2(y_train, np.dot(X_train, theta1))
                # J_test = R2(y_test, np.dot(X_test, theta1))
            elif j == numval - 1:
                X_test = X[j * g:, :]
                y_test = y[j * g:, :]
                X_train = X[:j * g, :]
                y_train = y[:j * g, :]
                theta1 = normalEqu_Reg(X_train, y_train, lamba)
                J_train = computeCost(X_train, y_train, theta1)
                J_test = computeCost(X_test, y_test, theta1)
                # J_train = R2(y_train, np.dot(X_train, theta1))
                # J_test = R2(y_test, np.dot(X_test, theta1))

            else:
                X_test = X[j * g:(j + 1) * g, :]
                y_test = y[j * g:(j + 1) * g, :]
                x1 = X[:j * g, :]
                x2 = X[(j + 1) * g:, :]
                y1 = y[:j * g, :]
                y2 = y[(j + 1) * g:, :]
                X_train = np.vstack([x1, x2])
                y_train = np.vstack([y1, y2])
                theta1 = normalEqu_Reg(X_train, y_train, lamba)
                J_train = computeCost(X_train, y_train, theta1)
                J_test = computeCost(X_test, y_test, theta1)
                # J_train=R2(y_train,np.dot(X_train,theta1))
                # J_test = R2( y_test, np.dot(X_test,theta1))
            J1.append(J_train)
            J4.append(J_test)
        J3.append(np.mean(J1))  # 每轮下来后对其取平均值当作这轮的值
        J6.append(np.mean(J4))
    return np.mean(J3), np.mean(J6)  # 总共完成后取平均值当作整个操作得出的最终代价函数值

def k_fold2(X,y,numval=10):
    """梯度下降法的k折交叉验证函数"""
    y=np.array(y)
    m=len(y)
    g=int(m/numval)
    J3=[]#创建装每次训练集代价函数的空列表
    J6=[]#创建装每次测试集代价函数的空列表
    for i in range(numval):
       X,y=random_data(X,y)#k次k折，每一次都会随机打乱顺序重排
       J1=[]#创建装一次中每轮训练集代价函数的空列表
       J4=[]#创建装一次中每轮测试集代价函数的空列表
       for j in range(numval):
           """分开训练集和验证集，得到各自的代价函数"""
           if j==0:

               X_test=X[:(j+1)*g,:]
               y_test=y[:(j+1)*g,:]
               X_train=X[(j+1)*g:,:]
               y_train=y[(j+1)*g:,:]
               theta1,J_train=gd.grad(X_train,y_train,0.003,1500)
               J_test=computeCost(X_test, y_test, theta1)
               #J_train = R2(y_train, np.dot(X_train, theta1))
               #J_test = R2(y_test, np.dot(X_test, theta1))
           elif j==numval-1:
               X_test = X[j*g:, :]
               y_test = y[j* g:, :]
               X_train = X[:j* g, :]
               y_train = y[:j*g, :]
               theta1, J_train = gd.grad(X_train, y_train, 0.003, 1500)
               J_test = computeCost(X_test, y_test, theta1)
               #J_train = R2(y_train, np.dot(X_train, theta1))
               #J_test = R2(y_test, np.dot(X_test, theta1))

           else:
               X_test = X[j*g:(j+1)*g,:]
               y_test = y[j*g:(j+1)*g,:]
               x1=X[:j*g,:]
               x2=X[(j+1)*g:,:]
               y1=y[:j*g,:]
               y2=y[(j+1)*g:,:]
               X_train=np.vstack([x1,x2])
               y_train=np.vstack([y1,y2])
               theta1, J_train = gd.grad(X_train, y_train, 0.003, 1500)
               J_test = computeCost(X_test, y_test, theta1)
               #J_train=R2(y_train,np.dot(X_train,theta1))
               #J_test = R2( y_test, np.dot(X_test,theta1))
           J1.append(J_train)
           J4.append(J_test)
       J3.append(np.mean(J1))#每轮下来后对其取平均值当作这轮的值
       J6.append(np.mean(J4))
    return np.mean(J3),np.mean(J6)#总共完成后取平均值当作整个操作得出的最终代价函数值