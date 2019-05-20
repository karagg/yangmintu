"""运用正规方程的简单线性回归函数"""
import numpy as np
import Nor_sta as ns
import PlotData as PD
import deal_o_n as dl
import pandas as pd
from computeCost import computeCost
import PlotData as PD
import hold_out as ht
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
print("数据初步可视化")
print(PD.plot1(X,y))#对数据进行初步可视化，看其标准化效果

X1=X.copy()
m=X.shape[0]#获取原特征矩阵的行数
n=X.shape[1]#获取原特征矩阵的列数
ones=np.ones(m).reshape(-1,1)
X=np.hstack([ones,X])#特征矩阵中合并一个x0矩阵，x0初始为1
#print(X)
theta=np.zeros((n+1,1))#theta初始为行数n+1的零矩阵
theta=ht.hold_out2(X,y,0.8,5)#留出集评估
print("一般情况下，J_train比较大，为过拟合，即高偏差情况，若J_test远大于J_train为欠拟合")
#theta=normalEqu(X,y)#选择较优值进行模型构建
#print('theta')
#print(theta)#显示theta

x_test1=np.linspace(0,1,300).reshape(-1,1)#显示拟合曲线查看拟合效果
ones2=np.ones((300,1)).reshape(-1,1)
x_test2=np.hstack([ones2,x_test1])#特征矩阵中合并一个x0矩阵，x0初始为1
y_pre1=np.dot(x_test2,theta)#通过theta,x,得到y矩阵
print("可视化拟合效果")
print(PD.plot2(X1,y,x_test1,y_pre1))#可视化拟合效果
