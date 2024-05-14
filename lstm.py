import torch
from torch import nn
import pytorch_lightning as pl
import torchvision.models as models

class LSTMModel(pl.LightningModule):
    def __init__(self):
        super(LSTMModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        self.conv = nn.Conv3d(in_channels=20,out_channels=32, kernel_size=(3,3,3) , padding=(1,1,1))
        # self.vgg = models.vgg16()
        # self.lstm = nn.LSTM(input_size=32*56*56, hidden_size=128, batch_first=True,bidirectional=True)
        self.lstm = nn.LSTM(input_size=3*224*224, hidden_size=128, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(128, 2)  # num_classes是分类类别数
        # self.fc2 = nn.Linear(64, 2)  # num_classes是分类类别数


    def forward(self, x):
        batch_size, clip_length, _, height, width = x.size()
        x = x.view(batch_size, clip_length, -1)  # 将输入数据展平成2-D张量
        # x = self.conv(x)

        print(x.shape)
        x = x.view(batch_size, clip_length, -1)
        _, (h_n, _) = self.lstm(x)

        out = h_n[-1]  # 取最后一个时间步的隐藏状态进行分类
        # out = self.fc2(out)

        # x = x.reshape(batch_size * clip_length, _, height, width)
        # x = self.layer1(x)
        # # print(self.res)
        # print(x.shape)
        # x = x.view(batch_size,clip_length,-1)
        # # print(x.shape)
        # _, (h_n, _) = self.lstm(x)
        # out = self.fc(h_n[-1])
        # out = self.fc2(out)
        return out

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.forward(x)
#         loss = nn.CrossEntropyLoss()(y_hat, y)
#         self.log("train_loss", loss)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.forward(x)
#         loss = nn.CrossEntropyLoss()(y_hat, y)
#         self.log("val_loss", loss)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
#
# # 创建数据加载器，假设为train_loader和val_loa

#
# model = LSTMModel()
# print(model)