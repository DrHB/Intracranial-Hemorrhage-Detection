import os
import cv2
import torch
import glob
import pydicom
import argparse

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import albumentations as albu
from matplotlib import pyplot as plt

from apex import amp
from tqdm import tqdm
from pydicom import dcmread
from torchvision import transforms
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensor
from sklearn.model_selection import GroupKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, HorizontalFlip, RandomBrightnessContrast

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True, help='name, logs saved under root/experiment_name/val_fold')
    parser.add_argument('--root', type=str, default='/home/xingjian/Code/RSNA/', help='Absolute path for the workspace')
    parser.add_argument('--dir_csv', type=str, default=None, help='contain train_with_meta.csv and test_with_meta.csv, else [root]/data')
    parser.add_argument('--dir_train_img', type=str, default=None, help='Where the training dcm are, else root/data/original_data/stage_1_train_images')
    parser.add_argument('--dir_test_img', type=str, default=None, help='Where the test dcm are, else root/data/original_data/stage_1_test_images')
    parser.add_argument('--cache_folder', type=str, default=None, help='Where to cache preprocessed images (windowing), else root//data/bsb_cache')
    parser.add_argument('--max_epochs', type=int, default=13)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_classes', type=int, default=6)
    parser.add_argument('--val_batch_size', type=int, default=64, help='to enable faster validation')
    parser.add_argument('--n_cpus', type=int, default=8)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--val_fold', type=int, required=True, help='which fold to use for validation')
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--model_name', type=str, required=True, help='efficientnet-b(0-7) or one of pretrainedmodels. Available models will be printed when error occurs')
    return parser.parse_args()

args = get_args()
root, dir_csv, dir_train_img, dir_test_img, cache_folder = args.root, args.dir_csv, args.dir_train_img, args.dir_test_img, args.cache_folder
gpu_idx, max_epochs, batch_size, val_batch_size, n_cpus = args.gpu_idx, args.max_epochs, args.batch_size, args.val_batch_size, args.n_cpus
num_folds, val_fold, n_classes = args.num_folds, args.val_fold, args.n_classes
experiment_name = args.experiment_name
model_name = args.model_name

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
torch.backends.cudnn.benchmark = True
from utils import criterion, toadaBN, TrainBatchSampler, TestBatchSampler, bsb_window, get_model
# Try loading the model before anything
model = get_model(model_name, n_classes)

if dir_csv is None:
    dir_csv = root+'data'
if dir_train_img is None:
    dir_train_img = root+'data/original_data/stage_1_train_images'
if dir_test_img is None:
    dir_test_img = root+'data/original_data/stage_1_test_images'
if cache_folder is None:
    cache_folder = root+'/data/bsb_cache'

if not os.path.exists(cache_folder):
    os.mkdir(cache_folder)
    print('Creating cache for pre-windows images, first epoch might take a long time')
else:
    print('Loading from cache folder')
# Save all models in 'saves' folder under cur_folder
ckpt_folder = os.path.join(root, experiment_name, 'fold'+str(val_fold))
if not os.path.exists(os.path.join(root, experiment_name)):
    print('First time running the experiment?')
    os.mkdir(os.path.join(root, experiment_name))
    os.mkdir(ckpt_folder)
elif not os.path.exists(ckpt_folder):
    os.mkdir(ckpt_folder)
print('Saving models in path:', ckpt_folder)

class IntracranialDataset(Dataset):
    def __init__(self, df, path, labels, transform=None):
        self.path = path
        self.data = df
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_name_ = os.path.join(cache_folder, self.data.loc[idx, 'Image'] + '.jpg')
            if not os.path.exists(img_name_):
                img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.dcm')
                img = dcmread(img_name)  
                img = bsb_window(img)
                plt.imsave(img_name_, img)
            img = plt.imread(img_name_)

            if self.transform:       
                augmented = self.transform(image=img)
                img = augmented['image']   

            if self.labels:
                labels = torch.tensor(
                    self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
                return {'image': img, 'labels': labels, 'idx':idx}    
            else:      
                return {'image': img, 'idx':idx}
        except:
            # print(self.data.loc[idx, 'Image'])
            new_idx = np.random.randint(len(self.data))
            return self.__getitem__(new_idx)

# Prep training dataframe
train = pd.read_csv(os.path.join(dir_csv, 'train_with_meta.csv')).drop_duplicates()
group_kfold = GroupKFold(n_splits=num_folds)
patients = train.patient.unique()
mapper = {}
for i, patient in enumerate(patients):
    mapper[patient] = i
patients = [mapper[patient] for patient in train.patient]
splits = [train.loc[train_idx] for _, train_idx in group_kfold.split(train, groups=patients)]
val, train = None, None
for i, df in enumerate(splits):
    if i == val_fold:
        val = df
    elif train is None:
        train = df
    else:
        train = pd.concat([train, df])
train_patients, val_patients = train.patient.unique(), val.patient.unique()
total_patients = list(train_patients)+list(val_patients)
print(len(total_patients)-len(set(total_patients)), 'overlapping between train & test patients')
print(len(train_patients), 'train patients;', len(val_patients), 'val patients')
train, val = train.reset_index(), val.reset_index()
print(len(train), 'training samples;', len(val), 'validation samples')

# Prep test dataframe
test = pd.read_csv(os.path.join(dir_csv, 'test_with_meta.csv'))
test = test[['Image', 'Label', 'patient', 'study']]
test.drop_duplicates(inplace=True)
test = test.reset_index()

transform_train = albu.Compose([
            Resize(224, 224),
            CenterCrop(200, 200),
            albu.HorizontalFlip(),
            albu.OneOf([
                albu.ShiftScaleRotate(shift_limit=.01, rotate_limit=20),
                albu.ElasticTransform(alpha_affine=10),
            ], p=1),
            albu.RandomBrightnessContrast(),
            albu.Normalize(p=1),
            ToTensor(),
        ])
transform_test = albu.Compose([
            Resize(224, 224),
            CenterCrop(200, 200),
            albu.HorizontalFlip(),
            albu.Normalize(p=1),
            ToTensor(),
        ])

train_dataset = IntracranialDataset(
    df=train, path=dir_train_img, transform=transform_train, labels=True)
val_dataset = IntracranialDataset(
    df=val, path=dir_train_img, transform=transform_test, labels=True)

test_dataset = IntracranialDataset(
    df=test, path=dir_test_img, transform=transform_test, labels=False)

train_sampler = TrainBatchSampler(train, batch_size)
val_sampler = TestBatchSampler(val, val_batch_size)
test_sampler = TestBatchSampler(test, val_batch_size)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=n_cpus)
data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=n_cpus)
# num_workers set to 0 to prevent some known error
data_loader_test = torch.utils.data.DataLoader(test_dataset,  batch_sampler=test_sampler, num_workers=0)


# Define the model here!
lr=1e-3
model = model.cuda()
model = toadaBN(model)

# lr /2 for pretrained, lr for head
op_params = []
for module in model.children():
    if isinstance(module, nn.Linear):
        print('Registering max_lr for fc layer (should only appear once)')
        argument = {'params': module.parameters(), 'lr': lr}
    else:
        argument = {'params': module.parameters(), 'lr': lr / 2}
    op_params.append(argument)
optimizer = optim.Adam(op_params)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(2*len(data_loader_train)))
model, optimizer = amp.initialize(model, optimizer, opt_level="O2",verbosity=0)

logs, val_logs = [], []
best_val_loss, ea_patience = 1e10, 0
for epoch in range(max_epochs):
    losses = []
    model.train()    
    tk0 = tqdm(data_loader_train)
    torch.backends.cudnn.benchmark = True
    for step, batch in enumerate(tk0):
        inputs, labels = batch["image"].cuda().float(), batch["labels"].cuda().float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        tk0.set_postfix({'loss':np.nanmean(losses)})
    logs.append(np.nanmean(losses))
    
    torch.backends.cudnn.benchmark = False
    tk0 = tqdm(data_loader_val)
    val_losses = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tk0):
            inputs, labels = batch["image"].cuda().float(), batch["labels"].cuda().float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            tk0.set_postfix({'val_loss':np.nanmean(val_losses)})
        val_logs.append(np.nanmean(val_losses))
    val_loss = val_logs[-1]
    print('Epoch {}/{}'.format(epoch, max_epochs - 1))
    print('Loss:{:.4f} val_loss: {:.4f}'.format(np.nanmean(losses), np.nanmean(val_losses)))
    if val_loss <= best_val_loss - .0005:
        ea_patience = 0
        best_val_loss = val_loss
        print('Better model, saving')
        torch.save(model.state_dict(), os.path.join(ckpt_folder, 'stage1_best.pth'))
        torch.save(model.state_dict(), os.path.join(ckpt_folder, 'stage1_'+str(epoch)+'.pth'))
        torch.save((args, logs, val_logs), os.path.join(ckpt_folder,'stage1_loss_stats.pth'))
    else:
        ea_patience += 1
        # If exceeds 2 cycles
        if ea_patience == 4:
            print('Early stopping triggered')
            break
# '''