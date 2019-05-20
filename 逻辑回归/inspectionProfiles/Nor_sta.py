# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:59:06 2019

@author: Lenovo
"""
import pandas as pd
import numpy as np
class Nor(object):
    """标准化和归一化"""
    def Dnormalize(self,data,columns,function,gp=False):
        """DataFrame类型归一化和标准化处理选择,默认qp=false,如果为true，则按其分组进行数据处理"""
        
        if columns not in data.columns:
            raise ValueError("invalid parameter: column '" + columns + "' doesn't exist.")
        if gp==True:
            return self.Gpstardlize(data,columns)
        is_int = data[columns].dtype == np.int8 or data[columns].dtype == np.int16 or data[columns].dtype == np.int32 or data[columns].dtype == np.int64
        is_uint = data[columns].dtype == np.uint8 or data[columns].dtype == np.uint16 or data[columns].dtype == np.uint32 or data[columns].dtype == np.uint64
        is_float = data[columns].dtype == np.float16 or data[columns].dtype == np.float32 or data[columns].dtype == np.float64
        if is_int or is_uint or is_float:
            data[columns] = data[columns].astype(np.float64)
        else:
            raise TypeError("invalid parameter: values in column '" + columns + "' should be numbers ")
        if function=="normalize":
            return self.Minmax_normalize(data,columns)
        elif function=="standard":
            return self.Standardlize(data,columns)
        else:
            raise ValueError("invalid parameter: function must be 'normalize' or 'standard'")
            
    def Minmax_normalize(self,data, columns):
        """归一化处理"""
        maxi=np.max(data[columns])#利用np的方法求最大值和最小值
        mini=np.min(data[columns])
        if maxi == mini:#异常处理，排查最大值与最小值相等的情况
            raise ValueError("invalid parameter value: maximum element equals to minimum element in the '" + columns + "' column.")
        else:
            dy = maxi - mini#运用广播进行归一化处理
            data[columns] = ( data[columns] - mini ) / dy
        return data
    def Standardlize(self,data,columns):
        """标准化"""
        std = np.std(data[columns])#取绝对值
        if std == 0:#异常处理，排查标准差等于零的情况
            raise ValueError("invalid parameter: standard deviation is 0 in the '" + columns + "' column.")
        else:#数据标准化
            mean = sum(data[columns]) / len(data[columns])
            data[columns] = ( data[columns] - mean ) / std
        return data
    def Gpstardlize(self,data,columns):
        """数据按照columns进行分组。组内标准化"""
        if type(data)!= pd.DataFrame:
            raise TypeError("invalid parameter: type in data should be dataframe")
        df=data.groupby(columns).transform(lambda x:x-x.mean())
        return df
            
    def S_n_normalize(self,data,function):
        """numpy和series类型归一化和标准化处理选择"""
        is_int = data.dtype == np.int8 or data.dtype == np.int16 or data.dtype == np.int32 or data.dtype == np.int64
        is_uint = data.dtype == np.uint8 or data.dtype == np.uint16 or data.dtype == np.uint32 or data.dtype == np.uint64
        is_float = data.dtype == np.float16 or data.dtype == np.float32 or data.dtype == np.float64
        if is_int or is_uint or is_float:
            data = data.astype(np.float64)
        else:
            raise TypeError("invalid parameter: values in data should be numbers ")
        if function=="normalize":
            return self.S_Minmax_normalize(data)
        elif function=="standard":
            return self.S_Standardlize(data)
        else:
            raise ValueError("invalid parameter: function must be 'normalize' or 'standard'")
    def S_Minmax_normalize(self,data):
        """归一化处理"""
        maxi=np.max(data,axis=0)#利用np的方法求最大值和最小值
        mini=np.min(data,axis=0)
        if sum(maxi - mini)==0:#异常处理，排查最大值与最小值相等的情况
            raise ValueError("invalid parameter value: maximum element equals to minimum element ")
        else:
            dy = maxi - mini#运用广播进行归一化处理
            data= ( data- mini ) / dy
        return data
    def S_Standardlize(self,data):
        """标准化"""
        std = np.std(data,axis=0)#取绝对值
        if np.any(std==0) == True:#异常处理，排查标准差等于零的情况
            raise ValueError("invalid parameter: standard deviation is 0 ")
        else:#数据标准化
            mean = np.mean(data,axis=0)
            data = ( data - mean ) / std
        return data  