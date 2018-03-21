from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
import numpy as np

class optimization:
    '''
    用于参数优化，目前主要针对rf,gbdt,xgb
    '''
    def __init__(self,train_x, train_y):
        self.train_x=train_x
        self.train_y=train_y

    def optimization_gbdt(self):
        train_x=self.train_x
        train_y=self.train_y
        # learning_rate and n_estimators
        param_test1 = {'n_estimators': range(50, 200, 10)}
        gsearch1 = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                min_samples_split=300,
                                                min_samples_leaf=20,
                                                max_depth=8,
                                                max_features='sqrt',
                                                subsample=0.8,
                                                random_state=10),
            param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
        gsearch1.fit(train_x, train_y)
        print("%-15s %-15s %-15s " % ("n_estimators", "1/5", "finish"))
        # max_depth
        param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 100)}
        gsearch2 = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                n_estimators=gsearch1.best_params_['n_estimators'],
                                                min_samples_leaf=20,
                                                max_features='sqrt',
                                                subsample=0.8,
                                                random_state=10),
            param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
        gsearch2.fit(train_x, train_y)
        print("%-15s %-15s %-15s " % ("max_depth", "2/5", "finish"))
        # min_samples_split and min_samples_leaf
        param_test3 = {'min_samples_split': range(100, 1900, 200), 'min_samples_leaf': range(5, 101, 10)}
        gsearch3 = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                n_estimators=gsearch1.best_params_['n_estimators'],
                                                max_depth=gsearch2.best_params_['max_depth'],
                                                max_features='sqrt',
                                                subsample=0.8,
                                                random_state=10),
            param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
        gsearch3.fit(train_x, train_y)
        print("%-15s %-15s %-15s " % ("split&leaf", "3/5", "finish"))
        # max_features
        param_test4 = {'max_features': range(1, len(train_x.columns), 1)}
        gsearch4 = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                n_estimators=gsearch1.best_params_['n_estimators'],
                                                max_depth=gsearch2.best_params_['max_depth'],
                                                min_samples_leaf=gsearch3.best_params_['min_samples_leaf'],
                                                min_samples_split=gsearch3.best_params_['min_samples_split'],
                                                subsample=0.8,
                                                random_state=10),
            param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
        gsearch4.fit(train_x, train_y)
        print("%-15s %-15s %-15s " % ("max_features", "4/5", "finish"))
        # subsample
        param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
        gsearch5 = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                n_estimators=gsearch1.best_params_['n_estimators'],
                                                max_depth=gsearch2.best_params_['max_depth'],
                                                min_samples_leaf=gsearch3.best_params_['min_samples_leaf'],
                                                min_samples_split=gsearch3.best_params_['min_samples_split'],
                                                max_features=gsearch4.best_params_['max_features'],
                                                random_state=10),
            param_grid=param_test5, scoring='roc_auc', iid=False, cv=5)
        gsearch5.fit(train_x, train_y)
        print("%-15s %-15s %-15s " % ("subsample", "5/5", "finish"))
        parameters = {'learning_rate': 0.1,
                    'n_estimators': gsearch1.best_params_['n_estimators'],
                    'max_depth': gsearch2.best_params_['max_depth'],
                    'min_samples_leaf': gsearch3.best_params_['min_samples_leaf'],
                    'min_samples_split': gsearch3.best_params_['min_samples_split'],
                    'max_features': gsearch4.best_params_['max_features'],
                    'subsample': gsearch5.best_params_['subsample'],
                    'random_state': 10
                    }
        print(parameters)
        return parameters
    def optimization_rf(self):
        train_x=self.train_x
        train_y=self.train_y
        param_test1 = {'n_estimators': range(100, 300, 30)}
        gsearch1 = GridSearchCV(
            estimator=RandomForestClassifier(min_samples_split=300,
                                            min_samples_leaf=20,
                                            max_depth=8,
                                            max_features='sqrt',
                                            random_state=10,
                                            n_jobs=-1,
                                            oob_score=True),
            param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
        gsearch1.fit(train_x, train_y)
        print('finish grid search 1/4 : n_estimators',gsearch1.best_params_['n_estimators'])
        # max_depth
        param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 1001, 100)}
        gsearch2 = GridSearchCV(
            estimator=RandomForestClassifier(n_estimators=gsearch1.best_params_['n_estimators'],
                                            min_samples_leaf=20,
                                            max_features='sqrt',
                                            random_state=10,
                                            n_jobs=-1,
                                            oob_score=True),
            param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
        gsearch2.fit(train_x, train_y)
        print('finish grid search 2/4 : max_depth',gsearch2.best_params_['max_depth'])
        # min_samples_split and min_samples_leaf
        param_test3 = {'min_samples_split': range(100, 1900, 200), 'min_samples_leaf': range(5, 101, 10)}
        gsearch3 = GridSearchCV(
            estimator=RandomForestClassifier(n_estimators=gsearch1.best_params_['n_estimators'],
                                            max_depth=gsearch2.best_params_['max_depth'],
                                            max_features='sqrt',
                                            random_state=10,
                                            n_jobs=-1,
                                            oob_score=True),
            param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
        gsearch3.fit(train_x, train_y)
        print('finish grid search 3/4 : min_samples_split and min_samples_leaf',gsearch3.best_params_['min_samples_split'],gsearch3.best_params_['min_samples_leaf'])
        # max_features
        param_test4 = {'max_features': range(1, len(train_x.columns), 1)}
        gsearch4 = GridSearchCV(
            estimator=RandomForestClassifier(n_estimators=gsearch1.best_params_['n_estimators'],
                                            max_depth=gsearch2.best_params_['max_depth'],
                                            min_samples_leaf=gsearch3.best_params_['min_samples_leaf'],
                                            min_samples_split=gsearch3.best_params_['min_samples_split'],
                                            random_state=10,
                                            n_jobs=-1,
                                            oob_score=True),
            param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
        gsearch4.fit(train_x, train_y)
        print('finish grid search 4/4 : max_features',gsearch4.best_params_['max_features'])
        parameters = {
            'n_estimators': gsearch1.best_params_['n_estimators'],
            'max_depth': gsearch2.best_params_['max_depth'],
            'min_samples_leaf': gsearch3.best_params_['min_samples_leaf'],
            'min_samples_split': gsearch3.best_params_['min_samples_split'],
            'max_features': gsearch4.best_params_['max_features'],
            'random_state': 10
        }
        print(parameters)
        return parameters
    def optimization_xgb(self):
        train_x=self.train_x
        train_y=self.train_y
        param_test1={'n_estimators': range(100, 200, 10)}
        gsearch1 = GridSearchCV(
            estimator=XGBClassifier(
                max_depth = 6,
                learning_rate = 0.1,
                silent = True,
                objective = 'reg:logistic',
                nthread = 40,
                gamma = 0,
                min_child_weight = 1,
                max_delta_step = 0,
                subsample = 0.8,
                colsample_bytree = 0.7,
                colsample_bylevel = 1,
                reg_alpha = 0,
                reg_lambda = 1,
                scale_pos_weight = 1,
                seed = 1440,
                missing = None,),
            param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
        gsearch1.fit(train_x, train_y)
        print("%-15s %-15s %-15s " % ("n_estimators", "1/5", gsearch1.best_params_['n_estimators']))

        param_test2={'learning_rate': np.arange(0.01, 0.2, 0.01)}
        gsearch2 = GridSearchCV(
                estimator=XGBClassifier(
                    max_depth = 6,
                    n_estimators = gsearch1.best_params_['n_estimators'],
                    silent = True,
                    objective = 'reg:logistic',
                    nthread = 40,
                    gamma = 0,
                    min_child_weight = 1,
                    max_delta_step = 0,
                    subsample = 0.8,
                    colsample_bytree = 0.7,
                    colsample_bylevel = 1,
                    reg_alpha = 0,
                    reg_lambda = 1,
                    scale_pos_weight = 1,
                    seed = 1440,
                    missing = None,),
                param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
        gsearch2.fit(train_x, train_y)
        print("%-15s %-15s %-15s " % ("learning_rate", "2/5", gsearch2.best_params_['learning_rate']))

        param_test3={'max_depth': range(3, 10, 1),'min_child_weight': np.arange(0.1, 1, 0.1)}
        gsearch3 = GridSearchCV(
                estimator=XGBClassifier(
                    n_estimators = gsearch1.best_params_['n_estimators'],
                    learning_rate = gsearch2.best_params_['learning_rate'],
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
                    missing = None,),
                param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
        gsearch3.fit(train_x, train_y)
        print("%-15s %-15s %-15s " % ("max_depth", "3/5", gsearch3.best_params_['max_depth']))
        print("%-15s %-15s %-15s " % ("min_child_weight", "3/5", gsearch3.best_params_['min_child_weight']))

        param_test4={'learning_rate': np.arange(gsearch2.best_params_['learning_rate']-0.01, gsearch2.best_params_['learning_rate']+0.01, 0.001)}
        gsearch4 = GridSearchCV(
                estimator=XGBClassifier(
                    max_depth=gsearch3.best_params_['max_depth'],
                    n_estimators = gsearch1.best_params_['n_estimators'],
                    # learning_rates = gsearch2.best_params_['learning_rate'],
                    min_child_weight=gsearch3.best_params_['min_child_weight'],
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
                    missing = None,),
                param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
        gsearch4.fit(train_x, train_y)
        print("%-15s %-15s %-15s " % ("learning_rate", "4/5", gsearch4.best_params_['learning_rate']))
        parameters = {
            'n_estimators': gsearch1.best_params_['n_estimators'],
            'max_depth': gsearch3.best_params_['max_depth'],
            'min_child_weight': gsearch3.best_params_['min_child_weight'],
            'learning_rate': gsearch4.best_params_['learning_rate'],
            'random_state': 1440
        }
        print(parameters)
        return parameters