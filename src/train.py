# -*- coding: utf-8 -*-
"""3D Train 2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pamHWBYbDIn2a2sE_52OzqE8CyxSb3cv

## Import
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install git+https://github.com/shijianjian/EfficientNet-PyTorch-3D

!pip install torchio

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/MyDrive/data/3d data"
#!unzip -q "/content/drive/MyDrive/data/3d data/open.zip"

import h5py # .h5 파일을 읽기 위한 패키지
import random
import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from plotly.offline import iplot
from utils import EarlyStopping, SAM
from tqdm.auto import tqdm
import torchio as tio
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation, 
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    RandomSwap,
    RandomGhosting,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from typing import Sequence, Callable
from efficientnet_pytorch_3d import EfficientNet3D

from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""## Hyperparameter Setting"""

CFG = {
    'EPOCHS':50,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':128,
    'SEED':41,
    'PATIENCE':10,
}

"""## Fixed RandomSeed"""

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

"""## Data Pre-processing"""

all_df = pd.read_csv('./train.csv')
all_points = h5py.File('./train.h5', 'r')

train_df = all_df.iloc[:int(len(all_df)*0.8)]
val_df = all_df.iloc[int(len(all_df)*0.8):]

class CustomDataset(Dataset):
    def __init__(self, id_list, label_list, point_list, transforms= Sequence[Callable]):
        self.id_list = id_list
        self.label_list = label_list
        self.point_list = point_list
        self.transforms = transforms
    def __getitem__(self, index):
        image_id = self.id_list[index]
        
        # h5파일을 바로 접근하여 사용하면 학습 속도가 병목 현상으로 많이 느릴 수 있습니다.
        points = self.point_list[str(image_id)][:]
        image = self.get_vector(points)

        if self.label_list is not None:
            label = self.label_list[index]
            image=torch.Tensor(image).unsqueeze(0)
            if self.transforms is not None:
              image = self.transforms(image)
            return image, label
        else:
            image=torch.Tensor(image).unsqueeze(0)
            if self.transforms is not None:
              image = self.transforms(image)
            return image
    
    def get_vector(self, points, x_y_z=[16, 16, 16]):
        # 3D Points -> [16,16,16]
        xyzmin = np.min(points, axis=0) - 0.001
        xyzmax = np.max(points, axis=0) + 0.001

        diff = max(xyzmax-xyzmin) - (xyzmax-xyzmin)
        xyzmin = xyzmin - diff / 2
        xyzmax = xyzmax + diff / 2

        segments = []
        shape = []

        for i in range(3):
            # note the +1 in num 
            if type(x_y_z[i]) is not int:
                raise TypeError("x_y_z[{}] must be int".format(i))
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
            segments.append(s)
            shape.append(step)

        n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
        n_x = x_y_z[0]
        n_y = x_y_z[1]
        n_z = x_y_z[2]

        structure = np.zeros((len(points), 4), dtype=int)
        structure[:,0] = np.searchsorted(segments[0], points[:,0]) - 1
        structure[:,1] = np.searchsorted(segments[1], points[:,1]) - 1
        structure[:,2] = np.searchsorted(segments[2], points[:,2]) - 1

        # i = ((y * n_x) + x) + (z * (n_x * n_y))
        structure[:,3] = ((structure[:,1] * n_x) + structure[:,0]) + (structure[:,2] * (n_x * n_y)) 

        vector = np.zeros(n_voxels)
        count = np.bincount(structure[:,3])
        vector[:len(count)] = count

        vector = vector.reshape(n_z, n_y, n_x)

        return vector

    def __len__(self):
        return len(self.id_list)

'''
    tio.ToCanonical(),
    tio.RandomMotion(p=0.2),
    tio.RandomBiasField(p=0.3),
    tio.RandomNoise(p=0.5),
    tio.Resize((64,64,64)),
    tio.RandomAffine(degrees=45),
    ZNormalization(),
        tio.CropOrPad((64,64,48)),
    tio.RandomGhosting(),
    tio.RandomSwap(),
  '''

training_transform = tio.Compose([
    tio.ToCanonical(),
    tio.RandomMotion(p=0.2),
    tio.RandomFlip(axes=('LR',)),
    tio.RandomBiasField(p=0.3),
    tio.RandomNoise(p=0.5),
    tio.Resize((64,64,64)),
    tio.RandomAffine(degrees=60),
    ZNormalization(),

])
validation_transform = tio.Compose([
    tio.Resize((64,64,64)),
    ZNormalization(),
])

train_dataset = CustomDataset(train_df['ID'].values,
                              train_df['label'].values,
                              all_points,
                              transforms=training_transform)

train_loader = DataLoader(train_dataset, 
                          batch_size = CFG['BATCH_SIZE'], 
                          shuffle=True,
                          pin_memory=True,    
                          num_workers=4)

val_dataset = CustomDataset(val_df['ID'].values, 
                            val_df['label'].values,
                            all_points,
                            transforms=validation_transform)

val_loader = DataLoader(val_dataset,
                        batch_size=CFG['BATCH_SIZE'],
                        pin_memory=True,    
                        shuffle=False,
                        num_workers=4)

"""## Model Define"""

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()
        self.model = EfficientNet3D.from_name("efficientnet-b3", override_params={'num_classes': 10}, in_channels=1)
        #


    def forward(self,x):

        x = self.model(x)
        nn.init.xavier_normal_(x)
#        x = self.classifier(x)
        return x

"""## Train"""

use_amp = True
save_path='best_model.pt'
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
early_stopping = EarlyStopping(patience = CFG['PATIENCE'], verbose = True, path =save_path )

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    best_score = 0
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for data, label in tqdm(iter(train_loader)):
            data, label = data.float().to(device), label.long().to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(data), label).backward()
            optimizer.second_step(zero_grad=True)

            #optimizer.step()           
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
                       
            train_loss.append(loss.item())
        
        if scheduler is not None:
            scheduler.step()
            
        val_loss, val_acc = validation(model, criterion, val_loader, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss)}] Val Loss : [{val_loss}] Val ACC : [{val_acc}]')
        early_stopping(-val_acc, model)
        if early_stopping.early_stop:
          print("Early stopping")
          break

def validation(model, criterion, val_loader, device):
    model.eval()
    true_labels = []
    model_preds = []
    val_loss = []
    with torch.no_grad():
        for data, label in tqdm(iter(val_loader)):
            data, label = data.float().to(device), label.long().to(device)
            
            model_pred = model(data)
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
    
    return np.mean(val_loss), accuracy_score(true_labels, model_preds)

"""## Train


"""

model = BaseModel()
model.eval()
#optimizer = torch.optim.AdamW(params = model.parameters(), lr = CFG["LEARNING_RATE"])
base_optimizer = torch.optim.SGD  
optimizer = SAM(model.parameters(), base_optimizer, lr=0.001, momentum=0.9)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)

train(model, optimizer, train_loader, val_loader, scheduler, device)

