import numpy as np
import pandas as pd

class construction_featrues:
    '''
    用于计算衍生指标，输出衍生指标dataframe
    '''
    def __init__(self,data):
        self.data = data
    def cal_featrues(self):
        df = self.data
        df = df[(df['TARGET'] == 1) | (df['TARGET'] == 0)]
        # (3)processing
        # (3.1) date
        df['open_datetime'] = pd.to_datetime(df['OPEN_DATE'], format='%Y%m%d')
        df['valid_datetime'] = pd.to_datetime(df['FIRST_VALID_DATE'], format='%Y%m%d')
        df['open_day'] = [i.day for i in df['open_datetime']]
        df['valid_day'] = [i.day for i in df['valid_datetime']]
        df['o_v_days'] = np.int64((df['valid_date'] - df['open_date']))/np.int64(86400e9)

        # (3.2) prom
        df_dummy_comp = pd.get_dummies(df['PROM_TYPE'], prefix='PROM', prefix_sep='_')
        df = df.join(df_dummy_comp)

        # (3.3) false
        df['IS_1000'] = 0
        df['ASSET_CHG'] = df['ASSET_NOW'] - df['ASSET_TOP']
        df.loc[df['ASSET_TOP'] <= 1050, 'IS_1000'] = 1
