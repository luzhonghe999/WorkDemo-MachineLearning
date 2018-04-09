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
         #(3.2) asset
        df['asset_up1']=df['ASSET_NOW1']-df['ASSET_NOW2']
        df['asset_up2']=df['ASSET_NOW2']-df['ASSET_NOW3']
        df['asset_up']=df['asset_up1']+df['asset_up2']
        df['hold1']= df['STK_NOW1']+ df['PRD_NOW1']
        df['hold2']= df['STK_NOW2']+ df['PRD_NOW2']
        df['hold3']= df['STK_NOW3']+ df['PRD_NOW3']
        df['hold_chg1']=df['hold1']-df['hold2']
        df['hold_chg2']=df['hold2']-df['hold3']
        df['hold_chg']=df['hold1']-df['hold3']
        df['hold_chg_percent']=df['hold_chg']/(np.abs(df['hold3'])+1)
        df['hold_stk_cnt_chg']=df['STK_BS_CNT1']-df['STK_BS_CNT2']
        df['cash_flow1']= df['OUT_CASH1'] - df['IN_CASH1']
        df['cash_flow2']= df['OUT_CASH2'] - df['IN_CASH2']
        df['cash_flow']= df['cash_flow1']+df['cash_flow2']
        df['cash_cnt_in']= df['IN_CNT1'] - df['IN_CNT2']
        df['cash_cnt_out']= df['OUT_CNT1'] - df['OUT_CNT2']
        df['stk_flow']= df['STK_SELL_AMT1'] - df['STK_BUY_AMT1']
        df['stk_sell_all']=df['STK_SELL_AMT1'] +df['STK_SELL_AMT2']
        df['stk_buy_all']=df['STK_BUY_AMT1'] +df['STK_BUY_AMT2']
        df['login1'] = df['APP_LOGIN1'] +df['FY_LOGIN1']+df['ZHLC_LOGIN1']
        df['login2'] = df['APP_LOGIN2'] +df['FY_LOGIN2']+df['ZHLC_LOGIN2']
        df['login_up']= df['login1'] - df['login2']
        df['cash_in_per_login1']=df['IN_CNT1']/(df['login1'] +1)
        df['cash_out_per_login1']=df['OUT_CNT1']/(df['login1'] +1)
        df['cash_in_per_login2']=df['IN_CNT2']/(df['login2'] +1)
        df['cash_out_per_login2']=df['OUT_CNT2']/(df['login2'] +1)
        df['cash_in_per_login_chg']=df['cash_in_per_login1']-df['cash_in_per_login2']
        df['cash_out_per_login_chg']=df['cash_out_per_login1']-df['cash_out_per_login2']
        df['asset_top_flow']=df['ASSET_TOP1']-df['ASSET_TOP2']
        df['asset_chg_percent1']=(df['ASSET_NOW1']-df['ASSET_NOW2'])/(df['ASSET_NOW2']+1)
        df['asset_chg_percent2']=(df['ASSET_NOW2']-df['ASSET_NOW3'])/(df['ASSET_NOW3']+1)
        df['asset_chg_percent']=(df['asset_up1']+df['asset_up2'])/(df['ASSET_NOW3']+1)
        df['stk_percent1']=(df['STK_NOW1']+1)/(df['ASSET_NOW1']+1)
        df['stk_percent2']=(df['STK_NOW2']+1)/(df['ASSET_NOW2']+1)
        df['stk_percent3']=(df['STK_NOW3']+1)/(df['ASSET_NOW3']+1)
        df['stk_percent_chg1']=df['stk_percent1']-df['stk_percent2']
        df['stk_percent_chg2']=df['stk_percent2']-df['stk_percent3']
        df['in_out_cnt_times']=(df['IN_CNT1'] + df['IN_CNT2']+1)/(df['OUT_CNT1'] + df['OUT_CNT2']+1)
        df['in_cash_aseet_percent1']=(df['IN_CASH1'])/(df['IN_CASH1']+df['OUT_CASH1']+1)
        df['out_cash_aseet_percent1']=(df['OUT_CASH1'])/(df['IN_CASH1']+df['OUT_CASH1']+1)
        df['in_cash_aseet_percent2']=(df['IN_CASH2'])/(df['IN_CASH2']+df['OUT_CASH2']+1)
        df['out_cash_aseet_percent2']=(df['OUT_CASH2'])/(df['IN_CASH2']+df['OUT_CASH2']+1)
        df['asset_now_top1']=df['ASSET_NOW1']-df['ASSET_TOP1']
        df['asset_now_top2']=df['ASSET_NOW2']-df['ASSET_TOP2']
        df['stk_now_top1']=df['STK_NOW1']-df['STK_TOP1']
        df['stk_now_top2']=df['STK_NOW2']-df['STK_TOP2']
        df['prd_now_top1']=df['PRD_NOW1']-df['PRD_TOP1']
        df['prd_now_top2']=df['PRD_NOW2']-df['PRD_TOP2']
        df[df['AGE']>100].AGE=100
        df[df['AGE']<16].AGE=0
        df[df['S_AGE']>30].S_AGE=30
        df[df['S_AGE']<0].S_AGE=0
        df['out_asset_up1']=df['OUT_ASEET_NOW1']-df['OUT_ASEET_NOW2']
        df['out_asset_up2']=df['OUT_ASEET_NOW2']-df['OUT_ASEET_NOW3']
        df['out_asset_up']=df['out_asset_up1']+df['out_asset_up2']
        df['commi_all']=df['NET_COMMI_GJ1']+df['NET_COMMI_GJ1']
        df['commi_rate']=(df['NET_COMMI_GJ1']+df['NET_COMMI_GJ1'])/(df['stk_sell_all']+df['stk_buy_all']+1)
        df[df['commi_rate']<=0].commi_rate=0
        df['trade_days_dif']=df['TRADE_DAYS1']-df['TRADE_DAYS2']
        df['is_oper1']=df['OUT_CASH1']+df['IN_CASH1']+df['STK_BUY_AMT1']+df['STK_SELL_AMT1']
        df[df['is_oper1']>0].is_oper1=1
        df[df['is_oper1']<=0].is_oper1=0
        df['is_oper2']=df['OUT_CASH2']+df['IN_CASH2']+df['STK_BUY_AMT2']+df['STK_SELL_AMT2']
        df[df['is_oper2']>0].is_oper2=1
        df[df['is_oper2']<=0].is_oper2=0
        df['is_oper']=df['is_oper1']+df['is_oper2']
        df['profit_asset1']=df['PROFIT1']/(df['ASSET_NOW2']+1)
        df['profit_asset2']=df['PROFIT2']/(df['ASSET_NOW3']+1)
        df['profit_asset']=(df['PROFIT1']+df['PROFIT2'])/(df['ASSET_NOW3']+1)

        # (3.3) false
        del df['CUSTOMER_NO']
        del df['PROM_TYPE']
        print('df.shape',df.shape)
        print('Generate Finish')
        return df