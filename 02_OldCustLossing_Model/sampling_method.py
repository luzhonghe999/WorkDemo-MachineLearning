# -*- coding: utf-8 -*-
import numpy as np
import sys
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
"""
import sampling_method as sm

train_x,train_y=SM.Smote(train_x,train_y, n=440).over_sampling()

train_x,train_y=SM.under_sampling(train_x,train_y,100000)

"""
def under_sampling(train_x,train_y,undersample_n):
    train_x['TARGET'] = train_y
    sample_data = train_x[train_x['TARGET'] == 0].sample(n=undersample_n)
    unsample_data = train_x[train_x['TARGET'] == 1]
    data_train = pd.concat([sample_data, unsample_data])
    train_y = data_train['TARGET']
    train_x = data_train.drop(['TARGET'], 1)
    return train_x, train_y

class Smote:
    def __init__(self, X_train,y_train, n=100, k=5):
        self.X_train=X_train
        self.y_train = y_train
        self.samples = X_train[y_train == 1].values  # 少数样本
        self.n_samples, self.n_attrs = self.samples.shape  # [行数，特征数]
        self.n = n  # 过采样比例，500就是500%
        self.k = k  # 临近个数 注意当k>n/100时代表的含义
        self.newindex = 0
        self.synthetic = np.zeros((int(self.n_samples * n / 100), self.n_attrs))  # 生成零矩阵，[行数*倍数，特征数]

    def over_sampling(self):
        train_x_2 = pd.DataFrame(self._over_sampling())
        train_y_2 = np.zeros(len(train_x_2), int)
        train_y_2 = pd.DataFrame([x + 1 for x in train_y_2])
        train_x_2.columns = self.X_train.columns
        train_x = pd.concat([self.X_train, train_x_2])
        train_y = np.hstack((self.y_train.values, np.transpose(train_y_2.values)[0]))
        return train_x,train_y

    def _over_sampling(self):
        n = int(self.n / 100)  # 过采样倍数 和论文算法相比，没有考虑n<100的情况，按论文来说，当n<100时，采用随机采样
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)  # 生成少数样本最邻近模型
        print("%-15s %-15s %-15s " % ("NNmethod", "modeling", "completed"))
        for i in range(len(self.samples)):  # 按索引遍历
            nnarray = neighbors.kneighbors(self.samples[i].reshape(1, -1), return_distance=False)[0]  # 生成[距离矩阵]，
            # 计算每一个少数样本到它所在的少数样本集合的距离矩阵，这里只返回临近点的index
            self._populate(n, i, nnarray)  # 插值获得新点，输入几倍就获得几个点
            self.view_bar(i, len(self.samples))  # 输出当前完成的点数，以及新矩阵的长度
        print(" ")
        return self.synthetic

    def _populate(self, n, i, nnarray):
        for j in range(n):  # n:采样倍数，对于每一个样本simple[i]都遍历一遍n,即生成n倍的采样结果是每个样本贡献n个值
            nn = random.randint(0, self.k - 1)  # 生成0到k-1的一个整数，用于定位simple[i]周围的点
            dif = self.samples[nnarray[nn]] - self.samples[i]  # 用这个临近点simple[x]的每一个特征-simple[i]的每一个特征
            gap = random.random()  # 随机获得比例系数
            self.synthetic[self.newindex] = self.samples[i] + gap * dif  # simple[i]+差值*比例系数，得到新点，
            # 新点赋给synthetic
            self.newindex += 1  # 进行下一个点

    @classmethod
    def view_bar(cls, num, total):
        rate = num / total
        rate_num = int(rate * 40)
        r = '\r%s%s%d%% ' % ('▇' * rate_num, " " * (40 - rate_num), (rate_num + 1) * 2.5,)
        sys.stdout.write(r)
        sys.stdout.flush()
