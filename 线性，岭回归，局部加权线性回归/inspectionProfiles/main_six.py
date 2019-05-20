"""局部加权线性回归"""
import numpy as np
import PlotData as PD
import regression_lwlr as lw
from plot_lwlr import plot_lwlr
from hold_out import hold_out3
rng=np.random.RandomState(0)
X=10*rng.rand(120)
def model(x):
    y=2*x-5+rng.randn(120)+1.8*np.sin(3*x)
    return y
y=model(X)#随机产生120个数据
print(PD.plot1(X,y))#初步数据可视化
X1=X.copy()
y1=y.copy()
x1_test=X1[80:]#此处数据是作为数据可视化用的，数据可视化要用一维数组
y1_test=y1[80:]
x1_train=X1[:80]
y1_train=y1[:80]
#print(ya)
X=X.reshape(-1,1)
m=len(y)#获取原特征矩阵的行数
ones=np.ones(m).reshape(-1,1)
X=np.hstack([ones,X])#特征矩阵中合并一个x0矩阵，x0初始为1
print(X)#输出特征数组
y=y.reshape(-1,1)
"""对数据进行分割，暂时分三分之二为训练集，三分之一为测试集，设定多个k值，通过数据可视化查看拟合情况然后选取最后k值
   进行留出集验证评估模型各性能指标"""
X_train=X[:80,:]#训练集和测试集分割
y_train=y[:80,:]
X_test=X[80:,:]
y_test=y[80:,:]
k=[0.15,0.3,0.45,0.6,0.75,0.9]#设置多个k值
plot_lwlr(X_test,X_train,y_train,x1_test,x1_train,y1_test,y1_train,k)#拟合情况可视化
#y_pre=lw.lwlrTest(X_test,X_train,y_train,0.45)
hold_out3(X,y,0.8,10,0.45)#10次留出集验证，训练集占比百分之八十
#输出训练集和验证集代价函数，MAE,MAPE,MSE,RMSE，R2
print("一般情况下，J_train比较大，为过拟合，即高偏差情况，若J_test远大于J_train为欠拟合,由于之前已经进行拟合情况可视化并选取k值，所以这个模型的拟合情况是较好的")