import numpy as np
import plot_data as pl
import Nor_sta as ns
from gradientDesent import gradientDesent
from gradientDesent_Reg import grad
from  F1 import F1
import ROC_AUC as ra
from Accuray import Accuracy
from hold_out import hold_out
import  Confusion_Matrix as cm
from plotDecisionBoundary import plotDB
from computecost import sigmoid
data = np.loadtxt('ex2data1.txt',delimiter=",")#下载数据文件，此数据文件数组为array数组
#print(data)
X=data[:,0:-1]
h=ns.Nor()
#X=h.S_n_normalize(X,"standard")#数组标准化，数据符合正态分布
X=h.S_n_normalize(X,"normalize")#数组归一化，数据都在零和一之间
X1=X[:,0]
X2=X[:,1]
y=data[:,-1]
y1=y
y=y.reshape(-1,1)
#pl.plotData(X1,X2,y1)
m=X.shape[0]#获取原特征矩阵的行数
n=X.shape[1]#获取原特征矩阵的列数
ones=np.ones(m).reshape(-1,1)
X=np.hstack([ones,X])#特征矩阵中合并一个x0矩阵，x0初始为1
m=len(y)
"""分百分之八十数据作为训练集和验证集进行留出集验证，剩下为测试集进行模型性能准确率,F1，混淆矩阵的评估"""
q=int(m*0.8)
X_train=X[:q,:]
y_train=y[:q,:]
X_test=X[q:,:]
y_test=y[q:,:]
#print(X_train)
#print(X_test)
#theta,J=grad(X_train,y_train,0.3,1500,0.1)#梯度下降加正则化取得目标theta值
#theta,J=gradientDesent(X_train,y_train,0.3,1500)#梯度下降取得目标theta值
#print(theta)
#print(J)
theta=hold_out(X_train,y_train,0.8,num_val=10)#10次留出集评估取平均的theta值
plotDB(theta,X1,X2,y1)#决策边界加数据可视化
y_pre=sigmoid(np.dot(X_test,theta))
yP=sigmoid(np.dot(X,theta))
y_pre1=y_pre
y_pre1[y_pre<0.5]=0
y_pre1[y_pre>=0.5]=1#得到的y_pre1为最终预测标签
acc=Accuracy(y_pre1, y_test)#调用准确率函数
print('accuracy')
print(acc)
f1=F1(y_test, y_pre1)#调用函数
print("F1")
print(f1)
cm.Con_Mat(y_test,y_pre1)#调用混淆矩阵函数

roc=ra.ROC(y,yP,np.arange(0,1,0.05))#调用ROC函数
print(roc)
auc=ra.AUC(y,yP,np.arange(0,1,0.05))#调用AUC函数
print(auc)
