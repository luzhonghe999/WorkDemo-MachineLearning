#encoding=utf8
from pandas import DataFrame
import pandas as pd

class origin_data_preprocessing:
    '''
    用于读取原始数据后预处理数据，主要为检查数据缺失情况、异常情况
    主要计算指标：
    *缺失率
    *零值率
    *最大值
    *最小值
    *独立值个数
    '''
    def __init__(self,data):
        self.origin_data=data
    def cal_index(self,path):
        columns_list=list(self.origin_data.columns)
        data_columns_info = []
        for column_name in columns_list:
            cal_series = self.origin_data[column_name]
            x_type = type(cal_series[0])
            x_count = len(cal_series)
            na_count= len(cal_series[cal_series.isnull()])
            na_rate = round(len(cal_series[cal_series.isnull()])/(x_count+0.01),4)
            zero_count = len(cal_series[cal_series == 0])
            zero_rate = round(len(cal_series[cal_series == 0])/(x_count+0.01),4)
            try:
                x_max = max(cal_series.drop_duplicates().dropna())
                x_min = min(cal_series.drop_duplicates().dropna())
            except:
                x_max = max(pd.to_numeric(cal_series, errors='coerce'))
                x_min = min(pd.to_numeric(cal_series, errors='coerce'))
            duplicate_count = len(cal_series.drop_duplicates())
            column_info = [column_name,x_type,na_count,na_rate,zero_count,zero_rate,x_max,x_min,duplicate_count]
            print(column_info)
            data_columns_info.append(column_info)
        df = DataFrame(data_columns_info)
        df.columns = ['column_name','x_type','na_count','na_rate','zero_count','zero_rate','x_max','x_min','duplicate_count']
        df.to_csv(path)
    def fillna(self,fill_values):
        return self.origin_data.fillna(fill_values)