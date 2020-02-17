import torch
from torch import nn


class mlp(nn.Module):

    def __init__(self):

        super(mlp, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(347, 75),
            nn.BatchNorm1d(75),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(75, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(10, 1)

        )

    def forward(self, x):
        x = self.model(x)

        return x


def main():

    data = torch.randn(2, 331)
    net = MLP()

    out = net(data)

    print(out.shape)



if __name__ == '__main__':
    main()


