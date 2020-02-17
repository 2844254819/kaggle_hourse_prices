import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from MLP import mlp

batchsz = 73
lr = 5
epochs = 50

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# 先联合，然后对每一列归一化
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))


# 将不是字符串的列的索引取出来
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)

# 这一列虽然是数字，但要当字符串数据处理
all_features['MSSubClass'] = all_features['MSSubClass'].astype(str)
pd.get_dummies(all_features['MSSubClass'], prefix='MSSubClass')


# 对缺失值填充零
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 离散化 举个例子，假设特征MSZoning里面有两个不同的离散值RL和RM，
# 那么这一步转换将去掉MSZoning特征，
# 并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1。
# 如果一个样本原来在MSZoning里的值为RL，那么有MSZoning_RL=1且MSZoning_RM=0。
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]

train_features = torch.tensor(all_features[: int(n_train * 0.8)].values, dtype=torch.float)
valid_features = torch.tensor(all_features[int(n_train * 0.8):n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)

train_labels = torch.tensor(
    train_data[: int(n_train * 0.8)].SalePrice.values, dtype=torch.float).view(-1, 1)

valid_labels = torch.tensor(
    train_data[int(n_train * 0.8):n_train].SalePrice.values, dtype=torch.float).view(-1, 1)

train_dataset = TensorDataset(train_features, train_labels)
train_iters = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)

valid_dataset = TensorDataset(valid_features, valid_labels)
valid_iters = DataLoader(valid_dataset, batch_size=batchsz)

net = mlp()
print(net)

criteon = nn.MSELoss()
optimizer = optim.Adam(params=net.parameters(), lr=lr)

for epoch in range(epochs):

    for batch_id, (x, y) in enumerate(train_iters):

        logits = net(x)

        loss = criteon(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 1000 == 0:
            print('epoch:[{}/{}], loss:{:.4f}'.format(epoch, epochs, loss))

net.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for batch_id, (x, y) in enumerate(valid_iters):
        logits = net(x)
        correct += ((logits.log() - y.log()) ** 2).sum().float().item()
        total += len(y)

    rmse = np.sqrt(correct / total)

    print('rmse:', rmse)

preds = net(test_features).detach().numpy()
test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('./data/submission_mlp.csv', index=False)

"""
参考 https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.16_kaggle-house-price
结果： 0.13470 效果一般， 不如Adaboosting的0.12296
"""
