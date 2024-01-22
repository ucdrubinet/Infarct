import pickle
import torch
import torchvision
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from torchvision import transforms, utils
from PIL import Image

from sklearn.metrics import classification_report, plot_confusion_matrix, balanced_accuracy_score

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
        self.targets = self.gt
        
        self.transform = transform
        
    def grabTiles(self):
        im_paths = []
        gt = []
        
        inf_ct = 0
        heal_ct = 0
        bg_ct = 0
        
        for wsi in self.WSI:
            heal_ct_case = 0
            bg_ct_case = 0
            if wsi in self.cases:
                for cls in os.listdir(self.path+wsi):
                    for image in os.listdir(self.path+wsi+'/'+cls):
                        if cls == 'BG':# and bg_ct_case <= 300:
                            gt.append(0)
                            bg_ct += 1
                            bg_ct_case += 1
                            
                            im_path = self.path+wsi+'/'+cls+'/'+image
                            im_paths.append(im_path)
                        elif cls == 'Heal':# and heal_ct_case <= 300:
                            gt.append(1)
                            heal_ct += 1
                            heal_ct_case += 1
                            
                            im_path = self.path+wsi+'/'+cls+'/'+image
                            im_paths.append(im_path)
                        elif cls == 'Inf':
                            gt.append(2)
                            inf_ct += 1
                            
                            im_path = self.path+wsi+'/'+cls+'/'+image
                            im_paths.append(im_path)
                            
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
    
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_counts = self._get_class_counts()
        self.min_class_count = min(self.class_counts.values())
        self.indices_by_class = self._indices_by_class()

    def _get_class_counts(self):
        target = self.dataset.targets if hasattr(self.dataset, "targets") else self.dataset.labels
        return {class_id: sum(1 for label in target if label == class_id) for class_id in set(target)}

    def _indices_by_class(self):
        target = self.dataset.targets if hasattr(self.dataset, "targets") else self.dataset.labels
        return {class_id: [i for i, label in enumerate(target) if label == class_id] for class_id in set(target)}

    def __iter__(self):
        for _ in range(self.min_class_count):
            for class_id, indices in self.indices_by_class.items():
                yield indices.pop(torch.randint(0, len(indices), size=(1,)).item())

    def __len__(self):
        return self.min_class_count * len(self.class_counts)   
    
def randomInit(m):
    print("Model Randomly Initialized")
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
        

def fit(model, loss_fn, optimizer, train_loader,train_loader_eval, val_loader, num_epochs, scheduler = None, stat_count=100, device=None,PATH = './saved_models/resnet18_Inf.pt'):
    curr_model_score = -1
    loss_epoch = []
    losses = []
    train_epoch = []
    val_epoch = []
    
    f1_val_epoch = []
    f1_train_epoch = []
    bacc_val_epoch = []
    bacc_train_epoch = []

    if device is not None:
        model.to(device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    randomInit(model)

    if scheduler == None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)    
    
    num_steps = len(train_loader)
    
    # Iterate through all Epochs
    for epoch in range(num_epochs):
        if epoch != 0:
            scheduler.step()
            loss_epoch.append(statistics.mean(losses))
            train_epoch.append((total_train_correct/total_train))
            val_epoch.append((total_correct/total_val))
            losses = []
            
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
                losses.append(loss.item())
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
                val_labels_full = val_labels_full + labels.cpu().data.numpy().tolist()
                
                
            total_train = 0
            total_train_correct = 0
            train_predicted_full = []
            train_labels_full = []
            for data in train_loader_eval:
                images, labels = data[0].cuda(), data[1].cuda()
                outputs = model(images)

                _, train_predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                total_train_correct += (train_predicted == labels).sum().item()

                train_predicted_full = train_predicted_full + train_predicted.cpu().data.numpy().tolist()
                train_labels_full = train_labels_full + labels.cpu().data.numpy().tolist()
                
                
        print("END OF EPOCH")
        val_dict = classification_report(val_labels_full, val_predicted_full, labels=[0,1,2],output_dict=True)
        train_dict = classification_report(train_labels_full, train_predicted_full, labels=[0,1,2],output_dict=True)
        
        bacc_val = balanced_accuracy_score(val_labels_full,val_predicted_full)
        bacc_train = balanced_accuracy_score(train_labels_full,train_predicted_full)

        val0_score = val_dict['0']['f1-score'] + 0.0001
        val1_score = val_dict['1']['f1-score'] + 0.0001
        val2_score = val_dict['2']['f1-score'] + 0.0001
        
        train0_score = train_dict['0']['f1-score'] + 0.0001
        train1_score = train_dict['1']['f1-score'] + 0.0001
        train2_score = train_dict['2']['f1-score'] + 0.0001
        
        f1_val_epoch.append(val_dict['2']['f1-score'])
        f1_train_epoch.append(train_dict['2']['f1-score'])
        
        bacc_val_epoch.append(bacc_val)
        bacc_train_epoch.append(bacc_train)
        
        print("Validation Set: ")
        print("Model Score = ", (total_correct/total_val)*val0_score*val1_score*val2_score)
        print("Accuracy = ", (total_correct/total_val))
        print("bAccuracy = ", (bacc_val))
        print("inf F1 = ", (val_dict['2']['f1-score']))
        
        print("Train Set: ")
        print("Model Score = ", (total_train_correct/total_train)*train0_score*train1_score*train2_score)
        print("Accuracy = ", (total_train_correct/total_train))
        print("bAccuracy = ", (bacc_train))
        print("inf F1 = ", (train_dict['2']['f1-score']))
        

        if curr_model_score < bacc_val: 
            curr_model_score = bacc_val
            print("Model Checkpoint saved!")
            torch.save({'model_state_dict': model.state_dict()}, PATH)
        
    try:
        plt.plot(loss_epoch,label='Loss')
        plt.savefig('loss_plot.png')

        plt.plot(train_epoch,label='Train Acc')
        plt.plot(val_epoch,label='Val Acc')
        plt.savefig('acc_plot.png')
        
        plt.plot(bacc_train_epoch,label='Train bAcc')
        plt.plot(bacc_val_epoch,label='Val bAcc')
        plt.savefig('bacc_plot.png')
        
        plt.plot(bacc_train_epoch,label='Train Infarct F1')
        plt.plot(f1_val_epoch,label='Val Infarct F1')
        plt.savefig('f1_plot.png')
    except:
        pass

    return loss_epoch, train_epoch, val_epoch


def main():
    parser = argparse.ArgumentParser(description="Training parameters")

    # Define arguments with their default values
    parser.add_argument("--model", type=str, default='r18', help="Choose model: r18, swinv2, or wrn50.")
    parser.add_argument("--segmentation_tile_dir", type=str, default='seg_data/', help="Directory for patched images")
    parser.add_argument("--save_path", type=str, default='./saved_models/r18.pt', help="Where to save the model")

    parser.add_argument("--bs", type=int, default=5, help="Batch size")
    parser.add_argument("--ep", type=int, default=40, help="Epochs")
    parser.add_argument("--lr", type=int, default=0.01, help="Learning rate")

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the arguments as variables in your code
    print(f"Architecture for training: {args.model}")
    print(f"Segmentation Tile Directory: {args.segmentation_tile_dir}")
    print(f"Path to saved model after training: {args.save_path}")
    print(f"Batch size: {args.bs}")
    print(f"Epochs: {args.ep}")
    print(f"Learning rate: {args.lr}")

    MODEL = args.model
	SEGMENTATION_TILE_DIR = args.segmentation_tile_dir
	SAVE_PATH = args.save_path
	EPOCH = args.ep
	BS = args.bs
	LR = args.lr

	train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5962484, 0.5533902, 0.6344057],
                             std=[0.00056322647, 0.00075884577, 0.00043305202])
    ])


	test_transform = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize(mean=[0.5962484, 0.5533902, 0.6344057],
	                             std=[0.00056322647, 0.00075884577, 0.00043305202])
	    ])

	all_cases = os.listdir(SEGMENTATION_TILE_DIR)

	train_cases = []
	val_cases = ['NA5137-16_HE','NA5137-17_HE','NA5137-18_HE','NA5057-16_HE','NA5057-17_HE','NA5057-18_HE',
	             'NA5077-16_HE','NA5077-17_HE','NA5077-18_HE']

	test_cases = ['NA5090-16_HE','NA5090-17_HE','NA5090-18_HE','NA5051-16_HE','NA5051-17_HE','NA5051-18_HE',
	              'NA5146-16_HE','NA5146-17_HE','NA5146-18_HE','NA5063-16_HE','NA5063-17_HE','NA5063-18_HE',
	              'NA5089-16_HE','NA5089-17_HE','NA5089-18_HE']

	delete_cases = ['NA5116-16_HE','NA5116-17_HE','NA5116-18_HE']


	all_cases = list(set(all_cases) - set(delete_cases))


	train_cases = list(set(all_cases) - set(test_cases))
	train_cases = list(set(train_cases) - set(val_cases))

	print("Working on train set")
	trainset = Patches(SEGMENTATION_TILE_DIR,train_cases,train_transform)

	print("Working on validation set")
	valset = Patches(SEGMENTATION_TILE_DIR,val_cases,test_transform)


	batch_size = BS
	num_epochs = EPOCH

	loss_f = nn.CrossEntropyLoss()


	if MODEL == 'wrn50':
		model = torchvision.models.wide_resnet50_2(pretrained =True)
		model.fc = nn.Linear(2048, 3) 
	elif MODEL == 'swinv2':
		model = torchvision.models.swin_v2_t(weights='IMAGENET1K_V1')
		model.head = nn.Linear(768,3)
	elif MODEL == 'r18':
		model = torchvision.models.resnet18(pretrained =True)
		model.fc = nn.Linear(512, 3)
	else:
		print("Model name not recognized, using resnet18")
		model = torchvision.models.resnet18(pretrained =True)
		model.fc = nn.Linear(512, 3) 

	optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

	target = trainset.targets

	class_sample_count = np.array(
	    [len(np.where(target == t)[0]) for t in np.unique(target)])
	weight = 1. / class_sample_count
	samples_weight = np.array([weight[t] for t in target])


	sampler = torch.utils.data.sampler.WeightedRandomSampler(
	                weights=samples_weight,
	                num_samples=len(trainset), replacement=True)


	train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)

	train_loader_eval = train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(valset,batch_size=batch_size,shuffle=True)

	loss_epoch, train_epoch, val_epoch = fit(model, loss_f, optimizer, train_loader,train_loader_eval, val_loader, num_epochs,PATH = SAVE_PATH)


if __name__ == "__main__":
    main()