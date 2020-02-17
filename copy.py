# -*- coding:utf8 -*-
# @TIME : 2018/4/15 下午17:55
# @Author : yjfiejd
# @File : house_price_test_1.py

import pandas as pd
import numpy as np
import os

# import matplotlib.pyplot as plt

# 【1】读取数据
# import data

train_df = pd.read_csv('./data/train.csv', index_col=0)
test_df = pd.read_csv('./data/test.csv', index_col=0)
# print(train_df.head())
# print(test_df.head())
print(train_df.shape)
print(test_df.shape)

# 【2】合并数据(先把train_df中的label取出来)
y_train = np.log1p(train_df.pop("SalePrice"))
print(y_train.shape)
all_df = pd.concat((train_df, test_df), axis=0)

# 【3】正确化变量属性
# int -> str -> One-hot
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head()
all_dummy_df = pd.get_dummies(all_df)  # 列数明显增多

# 【4】处理缺失值,这里用均值填充
print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)
print(mean_cols)

all_dummy_df = all_dummy_df.fillna(mean_cols)
print(all_dummy_df.isnull().sum())

# 【5】标准分布：因为每个房子有很多特征，每个特征对衡量标准不一样，
# 比如房子面积200，100，房间数量为3，5个，因为房子面积该特征数值大于房间数量，但是并不能说明房子面积这个特征更重要，所以我们做标准化处理
numeric_cols = all_df.columns[all_df.dtypes != 'object']  # 找出所有的数值列，在原来的all_df中寻找出列号
print(numeric_cols)
print(len(numeric_cols))  # 一共有35列是数值的，对他们标准化处理：得到均值为0，标准差为1对标准正态分布数据

numeric_cols_mean = all_dummy_df.loc[:, numeric_cols].mean()
numeric_cols_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_cols_mean) / numeric_cols_std
print(all_dummy_df.head())

# 【6】建立模型
# 把训练集与测试集拿出来
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

dummy_train_df = dummy_train_df.fillna(mean_cols)
dummy_test_df = dummy_test_df.fillna(mean_cols)
dummy_train_df.head()

X_train = dummy_train_df.values
X_test = dummy_test_df.values  # 再次注意，测试集最后用
# y_train = np.log1p(train_df.pop("SalePrice")) #先用交叉验证，训练模型，选择合适对参数

# 【7】调包sklearn机器学习库
# 通过上一个基础篇，我们知道列rideg最优对参数对alpha=15左右
from sklearn.linear_model import Ridge

ridge = Ridge(15)

# Bagging: 投票原理
# Bagging把很多的小分类器放在一起，每个train随机的一部分数据，然后把它们的最终结果综合起来（多数投票制）
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score

# params = [1, 10, 15, 20, 25, 30, 40]
# test_scores = []
# for param in params:
#    clf = BaggingRegressor(n_estimators=param, base_estimator=ridge)
#    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=15, scoring='neg_mean_squared_error'))
#    test_scores.append(np.mean(test_score))

import matplotlib.pyplot as plt
# plt.plot(params, test_scores)
# plt.title("Bagging_n_estimator vs CV Error")
# plt.show()
# 使用Bagging前后CV_Error变化：0.135 -> 0.132


# Boosting：增强学习原理 (Adaboost)
# Boosting: Boosting比Bagging理论上更高级点，它也是揽来一把的分类器。但是把他们线性排列。下一个分类器把上一个分类器分类得不好的地方加上更高的权重，这样下一个分类器就能在这个部分学得更加“深刻”。

from sklearn.ensemble import AdaBoostRegressor

params = [10, 15, 20, 25, 30, 35, 40, 50]
test_scores = []
for param in params:
    clf = BaggingRegressor(n_estimators=param, base_estimator=ridge)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

plt.plot(params, test_scores)
plt.title("Boosting_n_estimator vs CV Error")
plt.show()
# 结果显示alpha=25时候，表现最好，error<0.132

# 再来一波牛逼对XGBoost....
# 后期补上

# 【8】导出数据
br = BaggingRegressor(n_estimators=10, base_estimator=ridge)  # 由图像知道当参数为10 error最小
br.fit(X_train, y_train)
y_br = np.expm1(br.predict(X_test))
data = pd.DataFrame({'Id': test_df.index, 'SalePrice': y_br})
data.to_csv('example.csv')