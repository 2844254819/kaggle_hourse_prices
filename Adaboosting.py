import pandas as pd
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from MLP import mlp

batchsz = 73
lr = 7
epochs = 50

train_df = pd.read_csv('./data/train.csv', index_col=0)
test_df = pd.read_csv('./data/test.csv', index_col=0)

# y_train = np.log1p(train_df.pop('SalePrice'))
y_train = train_df.pop('SalePrice')

all_df = pd.concat((train_df, test_df), axis=0)

all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head()
all_dummy_df = pd.get_dummies(all_df)

mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)

# 标椎化
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_cols_mean = all_dummy_df.loc[:, numeric_cols].mean()
numeric_cols_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols]
                                     - numeric_cols_mean) / numeric_cols_std

dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

X_train = dummy_train_df.values
X_test = dummy_test_df.values

from sklearn.linear_model import Ridge

ridge = Ridge(15)

from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

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

br = BaggingRegressor(n_estimators=25, base_estimator=ridge)  # 由图像知道当参数为10 error最小
br.fit(X_train, y_train)
y_br = np.expm1(br.predict(X_test))
submission_df = pd.DataFrame({'Id': test_df.index, 'SalePrice': y_br})
print(submission_df.head(10))
submission_df.to_csv('./data/Adaboosting.csv', index=False)



