"""岭回归函数"""
import numpy as np
import Nor_sta as ns
import PlotData as PD
import deal_o_n as dl
import pandas as pd
import valid_curve as vc
from gradDesent  import grad
from comCost import computeCost
from gradientdesent import gradientDesent
from grad_learnning import grad_learnning
from MAE import MAE
from MAPE import MAPE
import MSE_RMSE
from R2 import R2
from O_u_fit import O_u_fit
from mapFeature import mapFeature
from normalEqu_Reg import normalEqu_Reg
from regular_terms import reg_choose
data = np.loadtxt('ex1data2.txt',delimiter=",")#下载数据文件，此数据文件数组为array数组

#print(data)#显示数据
X=data[:,0:-1]#提取特征变量数组
y=data[:,-1]#提取标签数组，此处提取出来后会变一维
y1=y#一维数组
X1=data[:,0]#一维数组
X2=data[:,1]#一维数组
#print(X2)
#print(y)
y=y.reshape(-1,1)#对数组变形，使维数与特征变量数组相同，实际意义，房子的价钱
#print(y)
#print(sum(X))
#调用自己的数组标准化库进行操作
h=ns.Nor()
#X=h.S_n_normalize(X,"standard")#数组标准化，数据符合正态分布
X=h.S_n_normalize(X,"normalize")#数组归一化，数据都在零和一之间
#y=h.S_n_normalize(y,"normalize")#数组归一化，数据都在零和一之间
print(X)
X1=X[:,0]#一维数组，为房子的面积
X2=X[:,1]#一维数组，为房子的高度
#print("数据初步可视化")
print(PD.plot3(X1,X2,y.ravel()))#对数据进行初步可视化，看其标准化效果
X1=X1.reshape(-1,1)
X2=X2.reshape(-1,1)
X=X1*X2#房子的总体积
X=vc.Degree(4,X)#多项式选择
m=len(y)
#ones=np.ones(m).reshape(-1,1)
#X=np.hstack([ones,X])#特征矩阵中合并一个x0矩阵，x0初始为1
#print(X)
"""对惩罚项运用交叉验证得出的代价函数值进行选择，选择出较好的惩罚值的模型，然后进行模型性能评估"""
reg_choose(X,y,numval=10)#10折交叉验证得到不同惩罚值的代价函数表
print("从函数表可以看出，J值均比较大，可能为欠拟合")
theta=normalEqu_Reg(X,y,0.0)#选出较好的theta，此时惩罚值为零
x_test1=np.linspace(0,1,300).reshape(-1,1)#显示拟合曲线查看拟合效果
x_test3=x_test1*x_test1
x_test3=vc.Degree(4,x_test3)
#print(x_test3)
y_pre1=np.dot(x_test3,theta)#通过theta,x,得到y矩阵
print("可视化拟合效果")
print(PD.plot4(X1,X2,y.ravel(),x_test1,x_test1,y_pre1.ravel()))#可视化拟合效果
y_pre=np.dot(X,theta)
#用较好的模型进行性能评估
MAE=MAE(y,y_pre)#调用MAE函数
MAPE=MAPE(y,y_pre)#调用MAPE函数
R2=R2(y,y_pre)#调用R2函数
MSE=MSE_RMSE.MSE(y,y_pre)#调用MSE函数
RMSE=MSE_RMSE.RMSE(y,y_pre)#调用RMSE函数
dr=pd.Series([MAE,MAPE,MSE,RMSE,R2],index=["MAE","MAPE","MSE","RMSE","R2"])#创立含有五种评估的矩阵
print(dr)#输出评估矩阵