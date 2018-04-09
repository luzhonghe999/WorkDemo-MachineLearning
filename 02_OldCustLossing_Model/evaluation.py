# -*- coding: utf-8 -*-
from sklearn.metrics import precision_recall_curve
from pandas import DataFrame
import numpy as np
import xlwt
import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

"""
import Evaluation as EV

time = time.strftime("%Y%m%d%H%M%S", time.localtime())
path = 'result'  + str(time) + '.xls'
ev=EV.evaluation(model, train_x, train_y, test_x, test_y)
ev.to_xls(path, n=30)
"""
class evaluation:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        '''
        class initial
        :param model: classification
        :param X_train: `
        :param y_train: `
        :param X_test: `
        :param y_test: `
        '''
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_pred_p = model.predict_proba(X_test)[:, 1]

    def set_style(self, name, height, bold=False, ci=0, fc=1):
        '''
        set excel cell style
        :param name: font name
        :param height: `
        :param bold:`
        :param ci: 0:black
        :param fc: 1:white
        :return: style
        '''
        style = xlwt.XFStyle()
        font = xlwt.Font()
        font.name = name
        font.bold = bold
        font.colour_index = ci
        font.height = height
        if fc != 1:
            pattern = xlwt.Pattern()
            pattern.pattern = xlwt.Pattern.SOLID_PATTERN
            pattern.pattern_fore_colour = fc
            style.pattern = pattern
        style.font = font
        return style

    def to_xls(self, path, n=30):
        '''
        save to excel
        :param path: save path
        :param n: feature importance number
        :return: null
        '''
        style1 = self.set_style('Time New Roman', 220, bold=True, ci=1, fc=0)
        style2 = self.set_style('Time New Roman', 220, False)
        f = xlwt.Workbook()
        sheet1 = f.add_sheet('basic_info', cell_overwrite_ok=True)
        sheet1.write(0, 0, 'date', style1)
        sheet1.write(1, 1, time.strftime("%Y-%m-%d", time.localtime()), style2)
        sheet1.write(1, 2, time.strftime("%H:%M:%S", time.localtime()), style2)

        sheet1.write(3, 0, 'shape', style1)
        sheet1.write(4, 1, self.X_train.shape[0], style2)
        sheet1.write(4, 2, self.X_train.shape[1], style2)

        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        sheet1.write(6, 0, 'confusion_matrix', style1)
        sheet1.write(7, 1, str(cm[0][0]), style2)
        sheet1.write(7, 2, str(cm[0][1]), style2)
        sheet1.write(8, 1, str(cm[1][0]), style2)
        sheet1.write(8, 2, str(cm[1][1]), style2)

        sheet1.write(10, 0, 'report', style1)
        report = classification_report(self.y_test, self.y_pred).replace('\n', ' ').split(' ')
        print(classification_report(self.y_test, self.y_pred))
        while '' in report:
            report.remove('')

        sheet1.write(11, 2, 'precision', style2)
        sheet1.write(11, 3, 'recall', style2)
        sheet1.write(11, 4, 'f1-score', style2)
        sheet1.write(11, 5, 'support', style2)
        for i in range(4, 9):
            sheet1.write(12,  i-3, report[i], style2)
        for i in range(9, 14):
            sheet1.write(13, i - 8, report[i], style2)
        sheet2 = f.add_sheet('pr_roc', cell_overwrite_ok=True)
        sheet2.write(0, 0, 'thresholds', style1)
        sheet2.write(0, 1, 'precision', style1)
        sheet2.write(0, 2, 'recall', style1)
        sheet2.write(0, 3, 'selectpercent', style1)
        sheet2.write(0, 4, 'f1score', style1)
        df = self._cal_precision_recall()
        for i in range(len(df)):
            j = 0
            while j < 5:
                if str(df.values[i, j]) == 'nan':
                    df.values[i, j] = 0
                sheet2.write(i + 1, j, round(df.values[i, j], 3), style2)
                j += 1

        # sheet2.write(0, 6, 'thresholds', style1)
        # sheet2.write(0, 7, 'fpr', style1)
        # sheet2.write(0, 8, 'tpr', style1)
        df, list1 = self._cal_roc_sk()
        # for i in range(len(df)):
        #     sheet2.write(i + 1, 6, df.values[i, 0], style2)
        #     sheet2.write(i + 1, 7, df.values[i, 1], style2)
        #     sheet2.write(i + 1, 8, df.values[i, 2], style2)
        sheet2.write(0, 10, 'thresholds_value', style1)
        sheet2.write(0, 11, 'roc_auc', style1)
        sheet2.write(0, 12, 'tpr_max', style1)
        sheet2.write(0, 13, 'fpr_max', style1)
        sheet2.write(1, 10, str(list1[0]), style2)
        sheet2.write(1, 11, str(list1[1]), style2)
        sheet2.write(1, 12, str(list1[2]), style2)
        sheet2.write(1, 13, str(list1[3]), style2)
        sheet3 = f.add_sheet('feature_importance', cell_overwrite_ok=True)
        df = self._cal_feature_importance(n=n)
        sheet3.write(0, 0, 'feature', style1)  # ,fc=18
        sheet3.write(0, 1, 'importance', style1)
        for i in range(len(df)):
            sheet3.write(i + 1, 0, df.values[i, 0], style2)
            sheet3.write(i + 1, 1, df.values[i, 1], style2)
        f.save(path)

    def _cal_precision_recall(self):
        '''
        write by lzh, different from sklearn precision_recall_curve
        thresholds step length can be controlled
        :return: dataframe
        '''
        ty = DataFrame(self.y_test)
        all_ty = len(ty)
        all_true = ty.apply(lambda x: x.sum())[0]
        df = DataFrame(np.vstack((self.y_test, self.y_pred_p)).transpose())
        df.rename(columns={0: 'test_y', 1: 'predict_prob'}, inplace=True)
        df = df.sort_values(by='predict_prob', ascending=False)
        t_precision = []
        t_recall = []
        select_percent = []
        thresholds = []
        i = 1
        while i >= 0:
            df_new = df[df.predict_prob >= i]
            select_true = df_new.apply(lambda x: x.sum())[0]
            select_all = len(df_new)
            t_precision.append((select_true+0.0001) / select_all)
            t_recall.append((select_true+0.0001) / all_true)
            select_percent.append(select_all / (all_ty + 0.00001))
            thresholds.append(i)
            i -= 0.005
        df = DataFrame()
        df['thresholds'] = np.array(thresholds)
        df['precision'] = np.array(t_precision)
        df['recall'] = np.array(t_recall)
        df['select_percent'] = np.array(select_percent)
        df['f1score'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])
        # print(df)
        return df

    def _cal_p_r_sk(self):
        '''
        sklearn precision_recall_curve
        :return: dataframe vary large
        '''
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_p)
        df = DataFrame()
        thresholds = thresholds.tolist()
        thresholds.append('1.1')
        df['thresholds'] = np.array(thresholds)
        df['precision'] = np.array(precision)
        df['recall'] = np.array(recall)
        df['f1score'] = 2 * df['precision']*df['recall']/(df['precision']+df['recall']+0.001)
        print(df)
        return df

    def _cal_roc_sk(self):
        '''
        sklearn roc curve
        :return: dataframe vary large
        '''
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_p)
        df = DataFrame()
        df['thresholds'] = np.array(thresholds)
        df['fpr'] = np.array(fpr)
        df['tpr'] = np.array(tpr)
        i = 0
        c_max = tpr[1] - fpr[1]
        thresholds_value = thresholds[1]
        tpr_max = tpr[1]
        fpr_max = fpr[1]
        while i < len(fpr):
            dist = tpr[i] - fpr[i]
            if dist > c_max:
                c_max = dist
                thresholds_value = thresholds[i]
                tpr_max = tpr[i]
                fpr_max = fpr[i]
            i += 1
        roc_auc = auc(fpr, tpr)
        return df, [thresholds_value, roc_auc, tpr_max, fpr_max]

    def _cal_feature_importance(self, n=30):
        '''
        creat feature importane
        :param n: select numbers of feature importance
        :return: dataframe
        '''
        feature_importance = DataFrame(list(zip(self.X_train.columns, self.model.feature_importances_)),
                                       columns=['col', 'f_importance'])
        df_f_impo = feature_importance.sort_values('f_importance', ascending=False)
        return df_f_impo.head(n)
