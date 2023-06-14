import os
import pickle
import torch
import torchvision
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from PIL import Image

from sklearn.metrics import classification_report, plot_confusion_matrix

import numpy as np
import statistics



class Patches(Dataset):
    """Non-infarcted WSI dataset"""

    def __init__(self, path, cases, transform=None):
        """
        Args:
            path (string): Path to the tile folder directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = path
        self.cases = cases
        self.WSI = os.listdir(path)
        
        self.im_paths, self.gt = self.grabTiles()
        
        self.transform = transform
        
    def grabTiles(self):
        im_paths = []
        gt = []
        
        inf_ct = 0
        heal_ct = 0
        bg_ct = 0
        
        for wsi in self.WSI:
            if wsi in self.cases:
                for cls in os.listdir(self.path+wsi):
                    for image in os.listdir(self.path+wsi+'/'+cls):
                        im_path = self.path+wsi+'/'+cls+'/'+image
                        im_paths.append(im_path)
                        if cls == 'BG':
                            gt.append(0)
                            bg_ct += 1
                        elif cls == 'Heal':
                            gt.append(1)
                            heal_ct += 1
                        elif cls == 'Inf':
                            gt.append(2)
                            inf_ct += 1
                            
        print("Total number of tiles: ")
        print("Heal = ", heal_ct, "| Inf = ", inf_ct, "| BG = ", bg_ct)
        
        return im_paths, gt

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.im_paths[idx]
        cls = self.gt[idx]
        
        #image = io.imread(img_name)
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
            
        sample = [image, cls]

        return sample
    
    
    
def randomInit(m):
    print("Model Randomly Initialized")
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
        

def fit(model, loss_fn, optimizer, train_loader, val_loader, num_epochs, scheduler = None, stat_count=100, device=None,PATH = './saved_models/resnet18_Inf.pt'):
    curr_model_score = -1
    loss_epoch = []
    train_epoch = []
    val_epoch = []

    if device is not None:
        model.to(device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    randomInit(model)

    if scheduler == None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)    
    
    num_steps = len(train_loader)
    
    # Iterate through all Epochs
    for epoch in range(num_epochs):
        if epoch != 0:
            scheduler.step()
            loss_epoch.append(f_loss.item())
            train_epoch.append((total_train_correct/total_train))
            val_epoch.append((total_correct/total_val))
            
        for train_ct in range(num_steps):           
            try:
                data = next(labelled_iter)
            except:
                labelled_iter = iter(train_loader)
                data = next(labelled_iter)

            with torch.enable_grad():
                model.train()
                images, labels = data[0].to(device,dtype=torch.float), data[1].to(device,dtype=torch.long)
                
                #print(labels)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                loss = loss_fn(outputs, labels)
                
                loss.backward()
                optimizer.step()
                # Print statistics on every stat_count iteration
                if (train_ct+1) % stat_count == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                                %(epoch+1, num_epochs, train_ct+1, 
                                len(train_loader), loss.item()))
                
        with torch.no_grad():
            model.eval()
            
            total_val = 0
            total_correct = 0
            val_predicted_full = []
            val_labels_full = []
            for data in val_loader:
                images, labels = data[0].cuda(), data[1].cuda()
                outputs = model(images)

                _, val_predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                total_correct += (val_predicted == labels).sum().item()

                val_predicted_full = val_predicted_full + val_predicted.cpu().data.numpy().tolist()
                val_labels_full = val_labels_full + val_labels_full.cpu().data.numpy().tolist()
                
                
            total_train = 0
            total_train_correct = 0
            train_predicted_full = []
            train_labels_full = []
            for data in train_loader:
                images, labels = data[0].cuda(), data[1].cuda()
                outputs = model(images)

                _, train_predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                total_train_correct += (train_predicted == labels).sum().item()

                train_predicted_full = train_predicted_full + train_predicted.cpu().data.numpy().tolist()
                train_labels_full = train_labels_full + train_labels_full.cpu().data.numpy().tolist()
                
                
        print("END OF EPOCH")
        val_dict = classification_report(val_labels_full, val_predicted_full, labels=[0,1,2],output_dict=True)
        train_dict = classification_report(train_labels_full, train_predicted_full, labels=[0,1,2],output_dict=True)

        val0_score = val_dict['0']['f1-score']
        val1_score = val_dict['1']['f1-score']
        val2_score = val_dict['2']['f1-score']
        
        train0_score = train_dict['0']['f1-score']
        train1_score = train_dict['1']['f1-score']
        train2_score = train_dict['2']['f1-score']
        
        print("Validation Set: ")
        print("Model Score = ", (total_correct/total_val)*val0_score*val1_score*val2_score)
        print("Accuracy = ", (total_correct/total_val))
        
        print("Train Set: ")
        print("Model Score = ", (total_train_correct/total_train)*train0_score*train1_score*train2_score)
        print("Accuracy = ", (total_train_correct/total_train))
        
        if curr_model_score < (total_correct/total_val)*val0_score*val1_score*val2_score*train2_score: 
            curr_model_score = (total_correct/total_val)*val0_score*val1_score*val2_score*train2_score
            print("Model Checkpoint saved!")
        torch.save({'model_state_dict': model.state_dict()}, PATH)
        
    try:
        plt.plot(loss_epoch,label='Loss')
        plt.savefig('loss_plot.png')

        plt.plot(train_epoch,label='Train Acc')
        plt.plot(val_epoch,label='Val Acc')
        plt.savefig('acc_plot.png')
    except:
        pass

    return loss_epoch, train_epoch, val_epoch


train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4409763317567454, 0.4016568471536302, 0.4988298669112181],
                             std=[0.31297803931100737, 0.2990562933047881, 0.33747493782548915])
    ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4409763317567454, 0.4016568471536302, 0.4988298669112181],
                             std=[0.31297803931100737, 0.2990562933047881, 0.33747493782548915])
    ])


SEGMENTATION_TILE_DIR = '/cache/S23_Infarct/seg_data_512/'
all_cases = os.listdir(SEGMENTATION_TILE_DIR)

train_cases = []
val_cases = ['NA5031-18_HE','NA5041-17_HE','NA5095-17_HE','NA5116-16_HE']

test_cases = ['NA-5029-16_HE','NA5093-17_HE','NA5091-16_HE','NA-5029-18_HE','NA5063-17_HE','NA5077-18_HE','NA5089-17_HE',
             'NA5095-16_HE','NA5114-16_HE','NA5146-17_HE','NA5077-17_HE','NA5146-18_HE','NA5051-17_HE','NA5007-17_HE']


train_cases = list(set(all_cases) - set(test_cases))
train_cases = list(set(train_cases) - set(val_cases))

print("Working on train set")
trainset = Patches(SEGMENTATION_TILE_DIR,train_cases,train_transform)

print("Working on validation set")
valset = Patches(SEGMENTATION_TILE_DIR,val_cases,test_transform)


batch_size = 75
num_epochs = 100

loss_f = nn.CrossEntropyLoss()

model = torchvision.models.resnet18()
model.fc = nn.Linear(512, 3)

optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size,shuffle=True)

loss_epoch, train_epoch, val_epoch = fit(model, loss_f, optimizer, train_loader, val_loader, num_epochs,PATH = './saved_models/resnet18_512.pt')

