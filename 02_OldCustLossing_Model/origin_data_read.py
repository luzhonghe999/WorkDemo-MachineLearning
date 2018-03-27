#encoding=utf8
import os
import pandas as pd
from pandas import DataFrame

class origin_data_read:
    '''
    用于读取文件夹下所有符合要求的文件并组合为datafarme
    指定文件类型为csv
    读取方式为GBK
    输入参数：文件夹路径、文件后缀、读取方式
    '''
    def __init__(self,data_folder,file_type='.csv',encoding_type='GBK'):
        self.data_folder=data_folder # 文件夹名称
        self.file_type=file_type
        self.encoding_type=encoding_type
        self.file_names=os.listdir(data_folder) # 读取文件夹下所有文件名
    def read_csv(self):
        data = DataFrame()
        for i in self.file_names:
            if os.path.splitext(i)[1]==self.file_type: # 选取csv文件进行组合
                df = pd.read_csv(self.data_folder+i, encoding=self.encoding_type)
                print(i,len(df))
                df['file_name']=i
                df_con=[data,df]
                data = pd.concat(df_con,ignore_index=True)
                print('df.shape',data.shape)
        print('Read Data Finish')
        return data