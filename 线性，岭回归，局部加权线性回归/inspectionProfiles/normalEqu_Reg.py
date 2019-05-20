import numpy as np
def normalEqu_Reg(X,y,lamba):
    """此函数通过加入正则化的正规方程求得当代价函数最小时的最小角，返回由最小theta排列组成的矩阵，其中X是特征矩阵，y是标签数组,lamba为正则化项"""
    n=np.shape(X)[1]#计算矩阵列数
    b=np.eye(n)
    b[0][0]=0#，由于theta0不参与，修改单位矩阵
    turn=np.linalg.pinv(np.dot(X.T,X)+lamba*b)#求XT*X加上单位岭矩阵的逆
    theta=np.dot(np.dot(turn,X.T),y)#求正规方程函数公式，也就是让代价函数的导数等于零(三维图中达到最凹点）时，theta的计算公式
    return theta#返回角度最佳矩阵