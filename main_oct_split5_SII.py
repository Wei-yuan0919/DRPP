'''
@Project : DeepLearning
@File    : main_oct_split.py
@IDE     : PyCharm
@Author  : Kyson. Li
@Date    : 2023/8/25 9:48
'''
import time

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.io import read_video
from PIL import Image
import torchvision.models as models
from random import randint
import pandas as pd
from utils.log_utils import save_label_pred_to_csv,save_label_pred_to_csv1,save_label_pred_to_csv2
import numpy as np
from datetime import datetime

#定义数据集

#分级标准  结肠癌 667, 肝移植  226， 胃癌 660，   肝癌 211
def load_label(ex_path=r'label_data/value_labels.xlsx', patient_id=282617):

    ex_data=pd.read_excel(ex_path)
    #读取excel中第一列的数据列表
    patient_ids=ex_data.iloc[:,0].values.tolist()
    #print(patient_ids)
    #print(len(patient_ids))
    #根据给定的patient中的数值,检索对应的第六列的数据
    patient_values=ex_data.iloc[:,1].values.tolist()
    #判定给定的patient_id是否在patient_ids中
    if patient_id in patient_ids:
        idx=patient_ids.index(patient_id)
        return patient_values[idx]
    else:
        return False
    # for id in range(len(patient_ids)):
    #     print(patient_ids[id])
    #     print(patient_values[id])


class Train_Dataset(Dataset):
    def __init__(self,data_dir,clip_length,transform=None):
        self.data_dir = data_dir
        self.clip_length = clip_length
        self.transform = transform
        self.identities = {'patient_id':[],'patient_dir':[]}
        self.video_list = []
        self.labels = []
        self._load_data()
    def _load_data(self):
       for root,dirs,files in os.walk(self.data_dir):
            for dir in dirs:
                if  dir.split('_')[1]=='A':
                    #print('dir：', dir)
                    patient_id=dir.split('_')[0]
                    print('patient_id：', patient_id)
                    video_dir=os.path.join(root,dir)
                    #video_dirs.append(video_dir)
                    self.identities["patient_id"].append(patient_id)
                    self.identities["patient_dir"].append(video_dir)
                    self.labels.append(load_label(patient_id=int(patient_id)))

       print('视频片段总数:', len(self.identities["patient_id"]))
       print('标签总数:', len(self.labels))
    def __len__(self):
        return len(self.identities["patient_id"])

    def __getitem__(self, index):
        patient_id=self.identities["patient_id"][index]
        video_dir=self.identities["patient_dir"][index]
        frames=[]
        #print('clip:',clip)
        for file in sorted(os.listdir(video_dir)):
            frame_path=os.path.join(video_dir,file)
            frames.append(frame_path)
        frame_count=len(frames)

        start_frame=randint(0,frame_count-self.clip_length)
        clip=frames[start_frame:start_frame+self.clip_length]


        frames_data = []
        # print('clip:',clip)
        for frame_path in clip:
            # print("frame_path:",frame_path)
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames_data.append(image)
        clip = torch.stack(frames_data)
        label = self.labels[index]
        if label <= 477.53:
            label = 0
        else:
            label = 1
        label = torch.tensor(label)
        # label = torch.tensor(label).float()
        return clip, label, patient_id

class VideoDataModule(pl.LightningDataModule):
    def __init__(self,data_dir,clip_length,batch_size,num_classes):
        super(VideoDataModule,self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.clip_length = clip_length
        self.num_classes = num_classes

    def setup(self,stage=None):
        transform=transforms.Compose([transforms.Resize((224, 224)),   #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomHorizontalFlip(p=0.5),
            # 随机旋转，范围从-10度到10度
            transforms.RandomRotation(degrees=10),

        ])
        dataset = Train_Dataset(self.data_dir, self.clip_length, transform=transform)
        print('dataset:',len(dataset))
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size


        if stage=='fit' or stage is None:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset,
                                                                                 [train_size, val_size])
        #     self.train_dataset=CustomDataset(self.data_dir, self.clip_length, transform=transform)
        #     self.val_dataset=CustomDataset(self.data_dir, self.clip_length, transform=transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=4)

class Model(pl.LightningModule):
    def __init__(self,num_classes,input_size=(10,3,224,224)):
        super(Model,self).__init__()
        self.max_accuracy = 0
        self.clip_length=clip_length
        self.num_classes=num_classes
        self.input_sze=input_size
        self.svg_dir=os.path.join('./log',datetime.now().strftime("%m%d%H%M"))

        # self.model=models.video.r3d_18(pretrained=False)
        # self.model = models.video.r3d_18(pretrained=True)
        self.model = models.video.mc3_18(pretrained=True)
        # mc3_18
        # self.model = models.video.r2plus1d_18(pretrained=True)

        self.model.stem[0] = nn.Conv3d(20, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        #prev_weight=self.model.stem[0].weight
        #new_weight=prev_weight.clone()
        #new_weight[:,:3,:,:,:]=prev_weight[:,:3,:,:,:]
        #self.model.stem[0].conv1=nn.Conv3d(10, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        #self.model.features[0].conv1=nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        #self.model.stem[0].weight=nn.Parameter(new_weight)
        # self.model.fc=nn.Linear(512,num_classes)
        self.model.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes)
        print("model",self.model)
        #self.flatten=nn.Flatten()

        self.train_sample = {'img_path': [], 'label_name': [], 'label': [], 'pred': [], 'pred_name': [], 'prob': [],
                             'features': []}
        self.test_sample = {'img_path': [], 'label_name': [], 'label': [], 'pred': [], 'pred_name': [], 'prob': [],
                            'features': []}

        self.outputs = {'train_label': [], 'train_prob': [], 'train_pred': [], 'test_label': [], 'test_prob': [],
                        'test_pred': [], }
        self.train_outs = {'train_label': [], 'train_prob': [], 'train_pred': [], }
        self.test_outs = {'test_label': [], 'test_prob': [], 'test_pred': []}

    def forward(self,x):
        features = self.model(x)
        output   = self.fc(features)
        #features=self.flatten(features)
        return output, features


    def training_step(self,batch,batch_idx):
        x,y,patient_id=batch
        #print('y',y)
        y_hat,features = self(x)
        _, predicted_labels = torch.max(y_hat, dim=1)
        accuracy = (predicted_labels == y).float().mean()
        # y_hat=y_hat.unsqueeze(1)
        criterion     = nn.CrossEntropyLoss()
        loss =  criterion(y_hat, y)
        # loss= nn.CrossEntropyLoss(y_hat,y)
        # loss=torch.nn.functional.mse_loss(y_hat,y)
        print('train_loss:',loss.item())
        print("train_acc",accuracy.item())
        self.log('train_loss',loss)
        self.log('train_acc', accuracy)
        return loss
    def validation_step(self,batch, batch_idx):
        x,y,patient_id=batch
        y_hat,features = self(x)
        _, predicted_labels = torch.max(y_hat, dim=1)
        score = y_hat[:, 1].cpu().numpy()  # 获取正类的预测分数


        # print("score",score)
        # print("y_hat",y_hat.shape)
        # print("y",y.shape)
        # print("y",y)
        # print("y_hat",y_hat)
        print("probs",score)
        criterion     = nn.CrossEntropyLoss()
        loss =  criterion(y_hat, y)
        accuracy = (predicted_labels == y).float().mean()
        # mae_loss = nn.CrossEntropyLoss(y_hat,y)
        # loss=torch.nn.functional.mse_loss(y_hat,y)
        # mae_loss=torch.nn.functional.l1_loss(y_hat,y)
        # print('mae_loss:',mae_loss.item())
        print('val_loss:', loss.item())
        print("accuarcy",accuracy.item())
        self.log('val_loss',loss)
        self.log('accuarcy',accuracy)
        # self.log('features',features)
        # self.log('val_mae',mae_loss)
        # 对字典sample中的图像信息进行更新
        #self.test_sample['img_path'].extend(img_info['img_path'])
        #self.test_sample['label_name'].extend(img_info['label_name'])
        self.test_sample['label'].extend(y.tolist())
        self.test_sample['pred'].extend(predicted_labels.cpu().detach().tolist())

        self.test_sample['patient_id'].extend(patient_id)
        self.test_sample['probs'].extend(score)

        # self.losses['mae_loss'].append(mae_loss.item())
        self.losses['val_loss'].append(loss.item())
        self.accuracy['accuracy'].append(accuracy.item())
        self.features['features'].append(features)


    def on_validation_epoch_start(self) -> None:
        self.test_sample = {'patient_id': [], 'label': [], 'pred': [],"probs":[]}
        self.losses={'val_loss':[]}
        self.accuracy={'accuracy':[]}
        self.features = {'features': []}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def on_validation_epoch_end(self):
            #val_loss = torch.stack([x['val_loss'] for x in self.outputs]).mean()
            #preds = torch.cat([x['preds'] for x in self.outputs])
            #labels = torch.cat([x['labels'] for x in self.outputs])

            #accuracy = (preds == labels).sum().item() / len(labels)
            #self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
            #self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)
            print('epoch validation end')
            sample_results={"testing_set":self.test_sample}
            val_loss = np.mean(self.losses['val_loss'])
            accuracy = np.mean(self.accuracy['accuracy'])


            print('————————————————————————————————————————————————————————————')
            print(len(sample_results['testing_set']['label']))
            print(len(sample_results['testing_set']['pred']))
            print(len(sample_results['testing_set']['patient_id']))
            print(len(sample_results['testing_set']['probs']))

            print("accuracy",accuracy)
            if  accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                tsne_features = torch.cat(self.features['features'], 0).cpu().data.numpy()
                np.save("tsne.npy", tsne_features)
            print('————————————————————————————————————————————————————————————')
            if not os.path.exists(self.svg_dir):
                os.makedirs(self.svg_dir)
            save_label_pred_to_csv2(label=sample_results['testing_set']['label'], patient_id=sample_results['testing_set']['patient_id'], pred=sample_results['testing_set']['pred'],probs = sample_results['testing_set']['probs'], acc= accuracy, mae=val_loss, current_epoch=self.current_epoch, save_dir=self.svg_dir)

def train(model,data_module,logger_=pl.loggers.TensorBoardLogger('logs/')):
    trainer=pl.Trainer(max_epochs=100,logger=logger_)
    trainer.fit(model,datamodule=data_module)

def test(model,data_module,logger_=pl.loggers.TensorBoardLogger('logs/')):
    trainer=pl.Trainer(logger=logger_)
    trainer.test(model,datamodule=data_module)



if __name__ == '__main__':
    print('start')
    print(load_label())
    data_dir= r'../data/data_YW'
    clip_length=20
    batch_size=8
    num_classes=2
    data_module=VideoDataModule(data_dir,clip_length,batch_size,num_classes) #[batch_size, clip_length, 3,224,224]
    model=Model(num_classes)
    train(model,data_module)
    #test(model,data_module)
# print('start')
# data_dir=r'.\data_YW'
# clip_length=100
# transform=transforms.Compose([ transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
# dataset = CustomDataset(data_dir, clip_length, transform=transform)
# print('dataset',dataset.__len__())