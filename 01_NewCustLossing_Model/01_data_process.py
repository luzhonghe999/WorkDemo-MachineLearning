from pandas import DataFrame
import origin_data_read as odr
import origin_data_preprocessing as odp
import construction_featrues as cf

import warnings
warnings.filterwarnings("ignore")
'''
数据预处理部分，主要包括：
数据读取、异常值检验、特征计算、特征异常值检验、数据分割等内容
'''

if __name__=="__main__":
    print('Reading Data ...')
    work_file=r'F:/WorkProjects/Data/WorkDemo(MachineLearning)/01_NewCustLossing_Data/'
    b=odr.origin_data_read(work_file + 'OriginData/',file_type='.csv',encoding_type='GBK') # 读取文件
    origin_data=b.read_csv()
    print('Detectionng Data Outlier ...')
    odp.origin_data_preprocessing(origin_data).cal_index(work_file + 'ModelFile/origin_data_info.csv') # 输出异常检验
    origin_data = odp.origin_data_preprocessing(origin_data).fillna(0) # 空值填充
    print('Generating Featrues ...')
    featrue_data = cf.construction_featrues(origin_data).cal_featrues() # 特征计算
    print('Detectionng Featrues Outlier ...')
    odp.origin_data_preprocessing(featrue_data).cal_index(work_file + 'ModelFile/featrue_data_info.csv') # 输出异常检验
    featrue_data = odp.origin_data_preprocessing(featrue_data).fillna(0) # 空值填充
    print('Saving Data ...')
    featrue_data.to_csv(work_file + 'FeatrueData/featrue_data.csv',index=False)
    df1=featrue_data[featrue_data['OPEN_DATE']<=20180131]
    del df1['OPEN_DATE']
    df1.to_csv(work_file + 'ModelData/model_data.csv',index=False)
    df2=featrue_data[featrue_data['OPEN_DATE']>20180131]
    del df2['OPEN_DATE']
    df2.to_csv(work_file + 'ModelData/test_data.csv',index=False)
    print('Success!')
