import numpy as np
import matplotlib.pyplot as plt
import regression_lwlr as lw
def plot_lwlr(X_test,X_train,y_train,x1_test,x1_train,y1_test,y1_train,k):
    """加权线性回归函数可视化"""
    plt.figure()

    for i in range(1,7):#六个子图，可视化不同k值时的数据拟合状况
        c = lw.lwlrTest(X_test, X_train, y_train,k[i-1])#调用局部加权回归算法得到预测值
        y_pre = c.ravel()
        j = np.argsort(x1_test)#将预测值根据特征数组大小排序来进行排序
        plt.subplot(3,2,i)#创建子图
        plt.plot(x1_train, y1_train, 'o', color='blue', label='y_true')#训练集可视化
        plt.plot(np.sort(x1_test), y_pre[j], '-', color='red', label="y_pre")#测试集拟合曲线可视化
        
    plt.show()

