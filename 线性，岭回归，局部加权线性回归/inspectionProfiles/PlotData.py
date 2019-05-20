import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot1(x,y):#一组数据的可视化
    plt.plot(x,y,'o',color='blue',label='y_true')#显示真实值散点图
    plt.xlabel("X")#标签设置
    plt.ylabel("y")
    plt.legend(loc='best')#图例显示最佳位置
    plt.show()#显示图形
def plot2(x,y,x_test,y_pre):
    plt.plot(x, y, 'o', color='blue', label='y_true')
    plt.plot(x_test,y_pre,'-',color='red',label="y_pre")#显示预测值的曲线图
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()
def plot3(x,y,z):#两个特征，三维可视化
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.scatter3D(x,y,z,c=z,cmap='Blues')#显示三维图，轴为标签数组，散点颜色随其数值增加而变深
    ax.set_xlabel("X1")#标签设置
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    plt.show()
def plot4(X1,X2,y,x_test1,x_test2,y_pre1):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X1, X2, y, c=y, cmap='Blues')  # 显示三维图，轴为标签数组，散点颜色随其数值增加而变深
    ax.plot3D(x_test1,x_test2,y_pre1,'gray')#显示预测值的三维曲线图
    ax.set_xlabel("X1")  # 标签设置
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    plt.show()