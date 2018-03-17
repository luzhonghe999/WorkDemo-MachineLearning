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
        # (1) date
        df['open_datetime'] = pd.to_datetime(df['OPEN_DATE'], format='%Y%m%d')
        df['valid_datetime'] = pd.to_datetime(df['FIRST_VALID_DATE'], format='%Y%m%d')
        #开户日与有效日
        df['open_day'] = [i.day for i in df['open_datetime']]
        df['valid_day'] = [i.day for i in df['valid_datetime']]
        #开户到有效所用时间
        df['o_v_days'] = np.int64((df['valid_datetime'] - df['open_datetime']))/np.int64(86400e9)
        #首次购买产品为股票或理财
        df['firts_buy'] = 0
        df.loc[df['FIRST_CHG_DATE']<df['FIRST_PRD_DATE'],'firts_buy']=1
        df.loc[df['FIRST_CHG_DATE']>=df['FIRST_PRD_DATE'],'firts_buy']=2
        #是否购买股票理财产品
        df.loc[df['FIRST_CHG_DATE']+df['FIRST_PRD_DATE']==0,'firts_buy'] = 0
        #客户经理与推荐人
        df['have_manager'] = 0
        df.loc[df['MANAGER_NO'] != 0,'have_manager'] = 1
        df['have_tjr'] = 0
        df.loc[df['EMP_NO_TJR'] != 0,'have_tjr'] = 1
        #理财产品到期
        df['prd_expire'] = 0
        df.loc[df['EXPIRE_DATE'] <=df['CAL_DATE'],'prd_expire'] = 1
        #特殊活动
        df['sp_activity'] = 0
        df.loc[df['ACTIVITY_LV1_NAME']=='互联网春天','sp_activity'] = 1
        #特殊金额
        df['sp_cash'] = 0
        df.loc[df['IN_CASH'] == 1000, 'sp_cash'] = 3
        df.loc[df['IN_CASH'] == 10000, 'sp_cash'] = 2
        df.loc[df['IN_CASH'] == 50000, 'sp_cash'] = 1
        # (3.2) prom
        df_dummy_comp = pd.get_dummies(df['SUB_COMP_CODE'], prefix='comp', prefix_sep='_')
        df = df.join(df_dummy_comp)
        df = df[df['FIRST_VALID_DATE']<20180228] 
        del df['OPEN_DATE']
        del df['CAL_DATE']
        del df['MONI_DATE']
        del df['FIRST_VALID_DATE']
        del df['FIRST_CHG_DATE']
        del df['FIRST_PRD_DATE']
        del df['LAST_VH_DATE']
        del df['SUB_COMP_CODE']
        del df['BRANCH_CODE']
        del df['MANAGER_NO']
        del df['EMP_NO_TJR']
        del df['OPEN_ACTIVITY_ID']
        del df['ACTIVITY_LV1_NAME']
        del df['LV1_NAME']
        del df['LV2_NAME']
        del df['open_datetime']
        del df['valid_datetime']
        print(type(df))
        return df