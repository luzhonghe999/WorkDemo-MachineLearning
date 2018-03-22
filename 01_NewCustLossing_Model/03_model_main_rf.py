import time    
from sklearn import metrics    
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import optimization as op
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import evaluation as ev
import warnings
warnings.filterwarnings("ignore")

def read_data(model_file,test_file):    
    train = DataFrame(pd.read_csv(model_file))
    test =  DataFrame(pd.read_csv(test_file))
    train_y = train.TARGET
    train_x = train.drop('TARGET', axis=1)  
    train_x_columns=train_x.columns
    test_y = test.TARGET
    test_x = test.drop('TARGET', axis=1)  
    test_x_columns=test_x.columns
    #标准化、归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    test_x = min_max_scaler.transform(test_x)
    train_x =DataFrame(train_x)
    train_x.columns=train_x_columns 
    test_x =DataFrame(test_x)
    test_x.columns=test_x_columns 
    return train_x, train_y, test_x, test_y  


def cal_roc_curve(test_y, predict_prob):
    """
    calculate fpr: False Positive / All Negative;tpr: True Positive / All Positive
    :param test_y: truly value (target)
    :param predict_prob: model value (predict probability)
    :return: fpr, tpr, thresholds
    """
    fpr, tpr, thresholds = roc_curve(test_y, predict_prob)
    return fpr, tpr, thresholds

def plot_roc(fpr, tpr, thresholds, figure_no,Classifier,color_type):
    """
    plot roc curve
    :param fpr: False Positive / All Negative
    :param tpr: True Positive / All Positive
    :param thresholds: thresholds
    :param figure_no: figure position
    :return: null
    """
    i = 0
    c_max = tpr[1] - fpr[1]
    thresholds_value = thresholds[1]
    tpr_max = tpr[1]
    fpr_max = fpr[1]
    while i < len(fpr):
        dist = tpr[i]-fpr[i]
        if dist > c_max:
            c_max = dist
            thresholds_value = thresholds[i]
            tpr_max = tpr[i]
            fpr_max = fpr[i]
        i += 1
    roc_auc = auc(fpr, tpr)
    plt.sca(figure_no)
    plt.plot(fpr, tpr, lw=1, label=Classifier+' ROC (area = %0.2f)' % roc_auc, color=color_type)
    plt.plot([fpr_max, fpr_max], [fpr_max, tpr_max], '--', lw=1, color='gray',
             label='thresholds = %0.3f' % thresholds_value)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

if __name__ == '__main__':    
    model_file = "f:/WorkProjects/Data/WorkDemo(MachineLearning)/01_NewCustLossing_Data/ModelData/model_data.csv"
    test_file = "f:/WorkProjects/Data//WorkDemo(MachineLearning)/01_NewCustLossing_Data/ModelData//test_data.csv"
    # model_file = "../../Data/WorkDemo(MachineLearning)/01_NewCustLossing_Data/ModelData/model_data.csv"
    # test_file = "../../Data//WorkDemo(MachineLearning)/01_NewCustLossing_Data/ModelData//test_data.csv"
    train_x, train_y, test_x, test_y = read_data(model_file,test_file) 
    optimization = 'None'
    start_time = time.time() 
    optimization = 'GridSearchCV'
    parameters = {'n_estimators': 280,
                    'max_depth':7,
                    'min_samples_split':100,
                    'min_samples_leaf':15,
                    'max_features':27
                }
    # parameters = op.optimization(train_x, train_y).optimization_rf()
    # (5.2) model core
    Classifier_name = 'rf'
    # xgb.XGBRegressor
    model = RandomForestClassifier(
        n_estimators=parameters['n_estimators'],
        max_depth=parameters['max_depth'],
        min_samples_split=parameters['min_samples_split'],
        min_samples_leaf=parameters['min_samples_leaf'],
        max_features=parameters['max_features'],
        random_state=10,
        n_jobs=-1,
        oob_score=True)
    model.fit(train_x, train_y)
    plt.figure(figsize=(8, 7))
    ax1 = plt.subplot(111)  
    print('training took %fs!' % (time.time() - start_time))    
    predict = model.predict(test_x)
    predict_prob = model.predict_proba(test_x)
    fpr, tpr, thresholds = cal_roc_curve(test_y, predict_prob[:, 1])
    plot_roc(fpr, tpr, thresholds, ax1,Classifier_name,'blue')
    precision = metrics.precision_score(test_y, predict)
    print(metrics.confusion_matrix(test_y, predict))
    recall = metrics.recall_score(test_y, predict)
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    accuracy = metrics.accuracy_score(test_y, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))

    plt.sca(ax1)
    plt.plot([0, 1], [0, 1], '--', lw=1, color=(0.6, 0.6, 0.2), label='Normal')
    plt.show()

    pathk = './result_rf_' + '_'+ str(time.time()) + '.xls'
    ev=ev.evaluation(model, train_x, train_y, test_x, test_y)
    ev.to_xls(pathk, n=50)