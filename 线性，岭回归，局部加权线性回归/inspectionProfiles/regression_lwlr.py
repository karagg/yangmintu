import numpy as np
def lwlr(X_testp,X_train,y_train,k):
    """局部加权算法函数，输入X_testP为测试点，X_train为训练特征集，y_train为训练标签集，k时与缩减速度有关的参数"""
    m=len(y_train)#训练集样本总量
    weights=np.eye(m)#创建只含有对角元素的权重矩阵
    for j in range(m):
        cha=X_testp-X_train[j,:]
        weights[j,j]=np.exp(((np.dot(cha,cha.T))**0.5)/(-2.0*k**2))#权重大小以指数级衰减
    turn = np.linalg.pinv(np.dot(X_train.T,np.dot(weights,X_train)))  # 求XT*wX的逆
    theta =np.dot(turn,np.dot(X_train.T,np.dot(weights,y_train)))#算法根据各点赋予权重的大小，解出某个点较佳的theta值
    return np.dot(X_testp,theta)#返回某点预测值
def lwlrTest(X_test,X_train,y_train,k):
    """用于训练集中每个点调用lwlr的函数，输入测试集特征数组和训练集，和衰减参数k，返回测试集预测值"""
    m,n=np.shape(X_test)
    y_pre=np.zeros((m,1))
    for i in range(m):
        y_pre[i][0]=lwlr(X_test[i],X_train,y_train,k)
    return y_pre
