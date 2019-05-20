import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_data import plotData

def plotDB(theta,X1,X2,y):
    #决策边界可视化
    plot_x=np.linspace(np.min(X1),np.max(X1))#横轴，范围是特征值X1的最小最大值
    plot_y=(-1/theta[2])*(theta[1]*plot_x+theta[0])#纵轴，令t0x0+t1x1+t2x2=0,解出x2,为纵轴
    plt.plot(plot_x,plot_y,label='Decsion')#决策边界显示
    plt.scatter(X1[y == 0], X2[y == 0])  # 以x0为横轴，x1为纵轴显示不同分类点
    plt.scatter(X1[y == 1], X2[y == 1])
    plt.title("decision boundary")
    plt.show()