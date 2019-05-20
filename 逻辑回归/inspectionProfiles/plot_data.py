
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plotData(X1,X2,y):
    """数据可视化函数"""
    plt.figure()#建立新画布
    plt.scatter(X1[y==0],X2[y==0])#以x0为横轴，x1为纵轴显示不同分类点
    plt.scatter(X1[y==1],X2[y==1])
    plt.show()