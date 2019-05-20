# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:28:15 2019

@author: Lenovo
"""
import numpy as np
import pandas as pd
class Nan(object):
    """缺失值处理"""
    def Isnull(self,data,operate):
        """判断缺失值是否为空以及执行填充操作"""
       
        if type(data)==np.ndarray:#numpy数组转换成pandas数组,方便后面操作
            if data.ndim==1:
                data.reshape(-1,1)#接下来要进行shape的索引，一维数组无法执行此操作
            data=pd.Series(data)
            is_int = data.dtype == np.int8 or data.dtype == np.int16 or data.dtype == np.int32 or data.dtype == np.int64
            is_uint = data.dtype == np.uint8 or data.dtype == np.uint16 or data.dtype == np.uint32 or data.dtype == np.uint64
            is_float = data.dtype == np.float16 or data.dtype == np.float32 or data.dtype == np.float64
            is_object=data.dtype==object
            if is_int or is_uint or is_float or is_object:#异常处理，处理数据有字符的异常
                data = data.astype(np.float64)
            else:
                raise TypeError("invalid parameter: values in data should be numbers ")
        if type(data)==list:
            data=pd.Series(data)#将非pd数据类型的数据转换成pd型的，方便使用fillna功能
        if np.any(data.isnull())==True:
            if operate=="max":
                return self.Maxfill(data)
            elif operate=="min":
                return self.Minfill(data)
            elif operate=="mean":
                return self.Meanfill(data)
            elif operate=="mode":
                return self.Modefill(data)
            elif operate=="mid":
                return self.Midfill(data)
            else:#输入值不在范围内的异常处理
                raise ValueError("invalid parameter:input should be 'max or min or mean or mode or'mid")
        else:
            print("无缺失值，不需要填充")
    def Minfill(self,data):
        """最小值填充"""
        data_minfill=data.fillna(data.min())
        return data_minfill
    def Meanfill(self,data):
        """均值填充"""
        data_meanfill=data.fillna(data.mean())
        return data_meanfill
    def Maxfill(self,data):
        """最大值填充"""
        data_maxfill=data.fillna(data.max())
        return data_maxfill
    def Midfill(self,data):
        """中位数填充"""
        data_midfill=data.fillna(data.median())
        return data_midfill
    def Modefill(self,data):
        """众数填充"""
        t=data.mode()
        data_modefill=data.fillna(t.loc[0])
        return data_modefill
class Err(object):
    """异常值处理"""
    def Errvaldeal(self,data,columns,funtion):
        """求出异常值范围并异常值处理选择"""
        """s=data[columns]
        mean=s.mean()
        sigma = s.std()
        upper=mean+3*sigma
        lower=mean-3*sigma#运用正态分布准则求出上限以及下限"""#其中一种异常值判断方法，个人觉得不实用
        quartiles=np.percentile(data[columns],[25,50,75])
        sig=quartiles[2]-quartiles[0]
        lower=quartiles[0]-1.5*sig
        upper=quartiles[2]+1.5*sig
        if funtion=='mean':
            return self.Errmeanfill(data,columns,upper,lower)
        elif funtion=='max':
            return self.Errmaxfill(data,columns,upper,lower)
        elif funtion=='min':
            return self.Errminfill(data,columns,upper,lower)
        elif funtion=='mid':
            return self.Errmidfill(data,columns,upper,lower)
        elif funtion=='mode':
            return self.Errmodefill(data,columns,upper,lower)
        else:
            raise ValueError("invalid parameter: function must be 'mean','max','min','mid' or 'mode'")

    def Errmeanfill(self,data,columns,upper,lower):
        """平均值填充"""
        f=data[columns].mean()
        data[columns][data[columns]<lower]=f#利用切片和掩码比较对数据进行异常值筛选
        data[columns][data[columns]>upper]=f
        return data
    def Errmaxfill(self,data,columns,upper,lower):
        """最大值填充"""
        f=data[columns].max()
        data[columns][data[columns]<lower]=f
        data[columns][data[columns]>upper]=f
        return data
    def Errminfill(self,data,columns,upper,lower):
        """最小值填充"""
        f=data[columns].min()
        data[columns][data[columns]<lower]=f
        data[columns][data[columns]>upper]=f
        return data
    def Errmidfill(self,data,columns,upper,lower):
        """中位数填充"""
        f=data[columns].median()
        data[columns][data[columns]<lower]=f
        data[columns][data[columns]>upper]=f
        return data
    def Errmodefill(self,data,columns,upper,lower):
        """众数填充"""
        t=data[columns].mode()
        f=t.loc[0]
        data[columns][data[columns]<lower]=f
        data[columns][data[columns]>upper]=f
        return data
    def S_errvaldeal(self,data,funtion):
        """求出异常值范围并异常值处理选择"""
        quartiles = np.percentile(data, [25, 50, 75])
        sig = quartiles[2] - quartiles[0]
        lower = quartiles[0] - 1.5 * sig
        upper = quartiles[2] + 1.5 * sig
        """s=data
        mean=s.mean()
        sigma = s.std()
        upper=mean+3*sigma
        lower=mean-3*sigma#运用正态分布准则求出上限以及下限"""
        if funtion=='mean':
            return self.SErrmeanfill(data,upper,lower)
        elif funtion=='max':
            return self.SErrmaxfill(data,upper,lower)
        elif funtion=='min':
            return self.SErrminfill(data,upper,lower)
        elif funtion=='mid':
            return self.SErrmidfill(data,upper,lower)
        elif funtion=='mode':
            if type(data)==pd.Series:
                return self.SErrmodefill(data,upper,lower)
            else:
                return self.Npmodefill(data,upper,lower)
        else:
            raise ValueError("invalid parameter: function must be 'mean','max','min','mid' or 'mode'")

    def SErrmeanfill(self,data,upper,lower):
        """平均值填充"""
        f=data.mean()
        data[data<lower]=f#利用切片和掩码比较对数据进行异常值筛选
        data[data>upper]=f
        return data
    def SErrmaxfill(self,data,upper,lower):
        """最大值填充"""
        f=data.max()
        data[data<lower]=f
        data[data>upper]=f
        return data
    def SErrminfill(self,data,upper,lower):
        """最小值填充"""
        f=data.min()
        data[data<lower]=f
        data[data>upper]=f
        return data
    def SErrmidfill(self,data,upper,lower):
        """中位数填充"""
        f=data.median()
        data[data<lower]=f
        data[data>upper]=f
        return data
    def SErrmodefill(self,data,upper,lower):
        """众数填充"""
        t=data.mode()
        f=t.loc[0]
        data[data<lower]=f
        data[data>upper]=f
        return data
    def Npmodefill(self,data,upper,lower):
        """numpy数组众数填充"""
        f=max(data.bitcount)
        data[data<lower]=f
        data[data>upper]=f
        return data
    