"""一个多项式基函数的回归"""
import numpy as np
import pandas as pd
import PlotData as PD
import valid_curve as vc
import  gradientdesent2 as gd
from MAE import MAE
from MAPE import MAPE
import MSE_RMSE
from R2 import R2
def make_data(N,err=1.0,rseed=1):
    """随机抽样数据"""
    rng=np.random.RandomState(rseed)
    X=rng.rand(N,1)**2
    y=10-1./(X.ravel()+0.1)
    if err>0:
        y+=err*rng.randn(N)
    return X,y
X,y=make_data(40)#产生四十个样本
X1=X.copy()
#print(X)
print(PD.plot1(X,y))#对数据进行初步可视化，方便选择x次方的，还是x开方的函数
y=y.reshape(-1,1)


"""对degree进行选择，一连串的开方或次方函数用交叉验证得到不同的训练集和验证集的代价函数，通过比较得到较优的degree,在进行模型性能评估"""
degree=np.arange(1,7)
vc.linear(X,y,degree,5)#交叉验证，最后一个参数为所几折交叉验证
X=vc.Degree2(4,X)#得到个代价函数后，根据初步可视化选好的模型，这里选择开方函数，若有合适的，可选择次方函数Degree
theta,J=gd.grad(X,y,0.003,1500)#生成theta和代价函数J
#可视化预测曲线
x_test1=np.linspace(0,1,300).reshape(-1,1)
x_test2=vc.Degree2(4,x_test1)
y_pre1=np.dot(x_test2,theta)#通过theta,x,得到y矩阵
print("可视化拟合效果")
print(PD.plot2(X1,y,x_test1,y_pre1))#可视化拟合效果
y_pre=np.dot(X,theta)
#用较好的模型进行性能评估
MAE=MAE(y,y_pre)#调用MAE函数
MAPE=MAPE(y,y_pre)#调用MAPE函数
R2=R2(y,y_pre)#调用R2函数
MSE=MSE_RMSE.MSE(y,y_pre)#调用MSE函数
RMSE=MSE_RMSE.RMSE(y,y_pre)#调用RMSE函数
dr=pd.Series([MAE,MAPE,MSE,RMSE,R2],index=["MAE","MAPE","MSE","RMSE","R2"])#创立含有五种评估的矩阵
print(dr)#输出评估矩阵





