"""运用梯度下降的简单线性回归函数
   主要是一个学习效率的选择"""
import numpy as np
import Nor_sta as ns
import PlotData as PD
import deal_o_n as dl
import pandas as pd
from computeCost import computeCost
from gradientdesent import gradientDesent
from grad_learnning import grad_learnning
from MAE import MAE
from MAPE import MAPE
import MSE_RMSE
from R2 import R2
from O_u_fit import O_u_fit
data = np.loadtxt('ex1data1.txt',delimiter=",")#下载数据文件，此数据文件数组为array数组
#print(data)#显示数据
X=data[:,0:-1]#提取特征变量数组
y=data[:,-1]#提取标签数组，此处提取出来后会变一维
#print(X)
#print(y)
y=y.reshape(-1,1)#对数组变形，使维数与特征变量数组相同
#print(y_1)
#print(sum(X))
#调用自己的数组标准化库进行操作
h=ns.Nor()
#X=h.S_n_normalize(X,"standard")#数组标准化，数据符合正态分布
X=h.S_n_normalize(X,"normalize")#数组归一化，数据都在零和一之间
print(X)
print("数据初步可视化")#其过欠拟合分析主要是数据可视化看出来的
print(PD.plot1(X,y))#对数据进行初步可视化，看其标准化效果
"""发现有异常值，调用自己的异常值处理库对异常值进行处理
g=dl.Err()
X=g.S_errvaldeal(X,'mean')
print((PD.plot1(X,y)))#进行进一步可视化看处理效果"""
X1=X
m=X.shape[0]#获取原特征矩阵的行数
n=X.shape[1]#获取原特征矩阵的列数
ones=np.ones(m).reshape(-1,1)
X=np.hstack([ones,X])#特征矩阵中合并一个x0矩阵，x0初始为1
#print(X)
q=int(m*0.3)
#print(q)
X_train=X[q:,:]#选出样本中的百分之三十作验证集，其余为训练集
y_train=y[q:,:]#运用切片完成
X_val=X[:q,:]
y_val=y[:q,:]
X1_train=X1[q:,:]
X1_val=X1[:q,:]
print(PD.plot1(X1_val,y_val))
#print(X_train)
#print(X_val)
theta=np.zeros((n+1,1))#theta初始为行数n+1的零矩阵
d=grad_learnning(X_train,y_train,theta)#对学习效率进行一个初步的筛选
print(d)#通过数组显示，可以查看较优值
theta, J_history=gradientDesent(X_train,y_train,theta,0.1,1500)#选择较优值进行模型构建
print("theta")
print(theta)#显示theta
print("代价函数的变化过程")
print(J_history)#显示代价函数的变化过程
x_test1=np.linspace(0,1,300).reshape(-1,1)#显示拟合曲线查看拟合效果
ones2=np.ones((300,1)).reshape(-1,1)
x_test2=np.hstack([ones2,x_test1])#特征矩阵中合并一个x0矩阵，x0初始为1
y_pre1=np.dot(x_test2,theta)#通过theta,x,得到y矩阵
print("可视化拟合效果")
print(PD.plot2(X1_train,y_train,x_test1,y_pre1))#可视化拟合效果

y_val_pre=np.dot(X_val,theta)
MAE=MAE(y_val,y_val_pre)#调用MAE函数
MAPE=MAPE(y_val,y_val_pre)#调用MAPE函数
R2=R2(y_val,y_val_pre)#调用R2函数
MSE=MSE_RMSE.MSE(y_val,y_val_pre)#调用MSE函数
RMSE=MSE_RMSE.RMSE(y_val,y_val_pre)#调用RMSE函数
dr=pd.Series([MAE,MAPE,MSE,RMSE,R2],index=["MAE","MAPE","MSE","RMSE","R2"])#创立含有五种评估的矩阵
print(dr)#输出评估矩阵







