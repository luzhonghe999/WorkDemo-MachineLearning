from pandas import DataFrame
import origin_data_read as odr
import origin_data_preprocessing as odp

if __name__=="__main__":
    work_file=r'F:/WorkProjects/Data/WorkDemo(MachineLearning)/01_NewCustLossing_Data/'
    b=odr.origin_data_read(work_file + 'ModelData/',file_type='.csv',encoding_type='GBK') # 读取文件
    origin_data=b.read_csv()
    odp.origin_data_preprocessing(origin_data).cal_index(work_file + 'ModelFile/origin_data_info.csv') # 输出异常检验
    origin_data = odp.origin_data_preprocessing(origin_data).fillna(-1) # 空值填充
    
