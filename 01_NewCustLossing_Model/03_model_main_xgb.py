import time    
from sklearn import metrics    
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
import optimization as op
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")
def read_data(model_file,test_file):    
    train = DataFrame(pd.read_csv(model_file))
    test =  DataFrame(pd.read_csv(test_file))
    train_y = train.TARGET
    train_x = train.drop('TARGET', axis=1)  
    test_y = test.TARGET
    test_x = test.drop('TARGET', axis=1)  
    #标准化、归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    test_x = min_max_scaler.transform(test_x)
    return train_x, train_y, test_x, test_y  

if __name__ == '__main__':    
    model_file = "../../Data/WorkDemo(MachineLearning)/01_NewCustLossing_Data/ModelData/model_data.csv"    
    test_file = "../../Data/WorkDemo(MachineLearning)/01_NewCustLossing_Data/ModelData/test_data.csv"    
    train_x, train_y, test_x, test_y = read_data(model_file,test_file) 
    optimization = 'None'
    start_time = time.time() 
    # optimization = 'GridSearchCV'
    # parameters = {'n_estimators': 50,
    #             'max_depth': 15,
    #             'max_features': 7,
    #             'silent': True,
    #             'objective': 'reg:logistic',
    #             'learning_rate': 0.1,
    #             'nthread': -1,
    #             'gamma': 0,
    #             'min_child_weight': 1,
    #             'max_delta_step': 0,
    #             'subsample': 0.80,
    #             'colsample_bytree': 0.7,
    #             'colsample_bylevel': 1,
    #             'reg_alpha': 0,
    #             'reg_lambda': 1,
    #             'scale_pos_weight': 1,
    #             'seed': 1440,
    #             'missing': None}
    parameters = op.optimization(train_x, train_y).optimization_xgb()
    # (5.2) model core
    Classifier_name = 'XGboost'
    # xgb.XGBRegressor
    model = XGBClassifier(
                    max_depth=parameters['max_depth'],
                    n_estimators = parameters['n_estimators'],
                    learning_rate = parameters['learning_rate'],
                    min_child_weight=parameters['min_child_weight'],
                    silent = True,
                    objective = 'reg:logistic',
                    nthread = 40,
                    gamma = 0,
                    max_delta_step = 0,
                    subsample = 0.8,
                    colsample_bytree = 0.7,
                    colsample_bylevel = 1,
                    reg_alpha = 0,
                    reg_lambda = 1,
                    scale_pos_weight = 1,
                    seed = 1440,
                    missing = None)
    model.fit(train_x, train_y, eval_metric='rmse', verbose=True, eval_set=[(test_x, test_y)], early_stopping_rounds=100)
    print('training took %fs!' % (time.time() - start_time))    
    predict = model.predict(test_x)   
    predict_prob = model.predict_proba(test_x)   
    # fpr, tpr, thresholds = cal_roc_curve(test_y, predict_prob[:, 1])
    # plot_roc(fpr, tpr, thresholds, ax1,classifier,color_type[i]) 
    precision = metrics.precision_score(test_y, predict)    
    print(metrics.confusion_matrix(test_y, predict))
    recall = metrics.recall_score(test_y, predict)    
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))    
    accuracy = metrics.accuracy_score(test_y, predict)    
    print('accuracy: %.2f%%' % (100 * accuracy))     
