
import time    
from sklearn import metrics    
import pickle as pickle    
import pandas as pd  
from pandas import DataFrame
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# Multinomial Naive Bayes Classifier    
def naive_bayes_classifier(train_x, train_y):    
    from sklearn.naive_bayes import MultinomialNB    
    model = MultinomialNB(alpha=0.01)    
    model.fit(train_x, train_y)    
    return model    
    
    
# KNN Classifier    
def knn_classifier(train_x, train_y):    
    from sklearn.neighbors import KNeighborsClassifier    
    model = KNeighborsClassifier()    
    model.fit(train_x, train_y)    
    return model    
    
    
# Logistic Regression Classifier    
def logistic_regression_classifier(train_x, train_y):    
    from sklearn.linear_model import LogisticRegression    
    model = LogisticRegression(penalty='l2')    
    model.fit(train_x, train_y)    
    return model    
    
    
# Random Forest Classifier    
def random_forest_classifier(train_x, train_y):    
    from sklearn.ensemble import RandomForestClassifier    
    model = RandomForestClassifier(n_estimators=8)    
    model.fit(train_x, train_y)    
    return model    
    
    
# Decision Tree Classifier    
def decision_tree_classifier(train_x, train_y):    
    from sklearn import tree    
    model = tree.DecisionTreeClassifier()    
    model.fit(train_x, train_y)    
    return model    
    
    
# GBDT(Gradient Boosting Decision Tree) Classifier    
def gradient_boosting_classifier(train_x, train_y):    
    from sklearn.ensemble import GradientBoostingClassifier    
    model = GradientBoostingClassifier(n_estimators=200)    
    model.fit(train_x, train_y)    
    return model    
    
    
# SVM Classifier    
def svm_classifier(train_x, train_y):    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    model.fit(train_x, train_y)    
    return model    
    

# SVM Classifier using cross validation    
def svm_cross_validation(train_x, train_y):    
    from sklearn.grid_search import GridSearchCV    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model    
    
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
    model_file = "F:/WorkProjects/Data/WorkDemo(MachineLearning)/01_NewCustLossing_Data/ModelData/model_data.csv"    
    test_file = "F:/WorkProjects/Data/WorkDemo(MachineLearning)/01_NewCustLossing_Data/ModelData/test_data.csv"    
    thresh = 0.5    
    model_save_file = None    
    model_save = {}  

    f = open('F:/WorkProjects/Data/WorkDemo(MachineLearning)/01_NewCustLossing_Data/ModelFile/model_test.txt', 'w') 

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT','GBDT']    # ,'SVMCV','SVM'
    classifiers = {'NB':naive_bayes_classifier,     
                  'KNN':knn_classifier,    
                   'LR':logistic_regression_classifier,    
                   'RF':random_forest_classifier,    
                   'DT':decision_tree_classifier,    
                #   'SVM':svm_classifier,    
                # 'SVMCV':svm_cross_validation,    
                 'GBDT':gradient_boosting_classifier    
    }    
        
    print('reading training and testing data...', file = f)    
    train_x, train_y, test_x, test_y = read_data(model_file,test_file)    
    color_type=['r','y','m','c','b','orange','lime','darkred']
    i = 0
    plt.figure(figsize=(16, 7))
    ax1 = plt.subplot(121)  
    for classifier in test_classifiers:    
        print('******************* %s ********************' % classifier, file = f)    
        start_time = time.time()    
        model = classifiers[classifier](train_x, train_y)    
        print('training took %fs!' % (time.time() - start_time), file = f)    
        predict = model.predict(test_x)   
        predict_prob = model.predict_proba(test_x)   
        fpr, tpr, thresholds = cal_roc_curve(test_y, predict_prob[:, 1])
        plot_roc(fpr, tpr, thresholds, ax1,classifier,color_type[i])
        if model_save_file != None:    
            model_save[classifier] = model    
        precision = metrics.precision_score(test_y, predict)    
        print(metrics.confusion_matrix(test_y, predict), file = f)
        recall = metrics.recall_score(test_y, predict)    
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall), file = f)    
        accuracy = metrics.accuracy_score(test_y, predict)    
        print('accuracy: %.2f%%' % (100 * accuracy), file = f)     
        i+=1
    if model_save_file != None:    
        pickle.dump(model_save, open(model_save_file, 'wb'))  
    f.close() 
    plt.sca(ax1)
    plt.plot([0, 1], [0, 1], '--', lw=1, color=(0.6, 0.6, 0.2), label='Normal')
    plt.show()