import time, os, glob

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import numpy as np
from PIL import Image
from tqdm import tqdm
import copy
from torch.utils.data import Dataset


def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=th.bool)
    mask[indices] = False
    return tensor[mask]



def main():
    parser = argparse.ArgumentParser(description="Training parameters")

    # Define arguments with their default values
    parser.add_argument("--model", type=str, default='r18', help="Choose model: r18, swinv2, or wrn50.")
    parser.add_argument("--model_path", type=str, default='./saved_models/r18.pt', help="Saved model")
    parser.add_argument("--smaller_model_path", type=str, default='./saved_models/r18_small.pt', help="If multiple FOV, specify path to second saved model")
    parser.add_argument("--wsi_dir", type=str, default='../wsi_evaluation/inf_tiles/val/', help="Directory to retrieve the patches for heatmaps")
    parser.add_argument("--save_np_dir", type=str, default='./Infseg/', help="Where to save the numpy heatmap")

    parser.add_argument("--multiFOV", type=int, default=0, help="0 for single FOV, 1 for multiFOV")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--stride", type=int, default=64, help="Stride for heatmap generation")
    parser.add_argument("--img_size", type=int, default=3072, help="Tile size from wsi_dir")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use")


    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the arguments as variables in your code
    print(f"Architecture for training: {args.model}")
    print(f"WSI Tile Directory: {args.wsi_dir}")
    print(f"Path to saved model : {args.model_path}")
    print(f"Path to saved heatmaps : {args.save_np_dir}")

    if args.multiFOV == 1:
    	print(f"Path to smaller FOV saved model : {args.smaller_model_path}")
    
    print(f"Batch size: {args.bs}")
    print(f"Stride: {args.stride}")
    print(f"Patch size: {args.img_size}")
    print(f"Workers: {args.workers}")
    print(f"GPU: {args.gpu}")

    IMG_DIR  = args.wsi_dir
	MODEL_SEG_DIR = args.model_path
	MODEL_SEG_DIR_SMALL = args.smaller_model_path
	SAVE_NP_DIR = args.save_np_dir
	isMultiFOV = args.multiFOV


    MODEL = args.model
	img_size = args.img_size
	tile_size = int(args.img_size / 6)
	stride = args.stride
	batch_size = args.bs 
	num_workers = args.workers

	class HeatmapDataset(Dataset):
	    def __init__(self, tile_dir, row, col, stride=1):
	        """
	        Args:
	            tile_dir (string): path to the folder where tiles are
	            row (int): row index of the tile being operated
	            col (int): column index of the tile being operated
	            stride: stride of sliding 
	        """
	        self.tile_size = tile_size
	        self.img_size = img_size
	        self.stride = stride
	        padding = tile_size
	        large_img = torch.ones(3, 3*self.img_size, 3*self.img_size)
	        
	        for i in [-1,0,1]:
	            for j in [-1,0,1]:
	                img_path = tile_dir+'/'+str(row+i)+'/'+str(col+j)+'.jpg'
	                try:
	                    img = Image.open(img_path)
	                    img = transforms.ToTensor()(img) 
	                except:
	                    img = torch.ones(3,self.img_size, self.img_size)
	                
	                large_img[:, (i+1)*self.img_size:(i+2)*self.img_size,(j+1)*self.img_size:(j+2)*self.img_size] = img
	        
	        large_img = normalize(large_img)
	        
	        self.padding_img = large_img[:,self.img_size-padding:2*self.img_size+padding, self.img_size-padding:2*self.img_size+padding]
	        self.len = (self.img_size//self.stride)**2
	        
	    def __getitem__(self, index):

	        row = (index*self.stride // self.img_size)*self.stride
	        col = (index*self.stride % self.img_size)

	        img = self.padding_img[:, row:row+self.tile_size, col:col+self.tile_size]        
	    
	        return img

	    def __len__(self):
	        return self.len

	normalize = transforms.Normalize(mean=[0.5962484, 0.5533902, 0.6344057],
                             std=[0.00056322647, 0.00075884577, 0.00043305202])

	to_tensor = transforms.ToTensor()

	# Retrieve Files
	filenames = glob.glob(IMG_DIR + '*')
	filenames = [filename.split('/')[-1] for filename in filenames]
	filenames = sorted(filenames,reverse=False)
	print("Cases to be processed: ", filenames)


	# Check GPU:
	use_gpu = torch.cuda.is_available()

	if MODEL == 'wrn50':
		seg_model = torchvision.models.wide_resnet50_2(pretrained =True)
		seg_model.fc = nn.Linear(2048, 3) 
	elif MODEL == 'swinv2':
		seg_model = torchvision.models.swin_v2_t(weights='IMAGENET1K_V1')
		seg_model.head = nn.Linear(768,3)
	elif MODEL == 'r18':
		seg_model = torchvision.models.resnet18(pretrained =True)
		seg_model.fc = nn.Linear(512, 3)
	else:
		print("Model name not recognized, using resnet18")
		seg_model = torchvision.models.resnet18(pretrained =True)
		seg_model.fc = nn.Linear(512, 3) 

	checkpoint = torch.load(MODEL_SEG_DIR,map_location=torch.device('cuda:'+args.gpu))

	if isMultiFOV:
		seg_model_small = seg_model
		checkpoint_small = torch.load(MODEL_SEG_DIR_SMALL,map_location=torch.device('cuda:'+args.gpu))
		seg_model_small.load_state_dict(checkpoint_small['model_state_dict'])

	seg_model.load_state_dict(checkpoint['model_state_dict'])

	if use_gpu:
	    seg_model = seg_model.cuda(args.gpu)
	    seg_model_small = seg_model_small.cuda(args.gpu)
	else:
		print("Look into possible GPU issues") 


	for filename in filenames[:]:
	    print("Now processing: ", filename)
	    
	    # Retrieve Files
	    TILE_DIR = IMG_DIR+'{}/0/'.format(filename)

	    imgs = []
	    for target in sorted(os.listdir(TILE_DIR)):
	        d = os.path.join(TILE_DIR, target)
	        if not os.path.isdir(d):
	            continue

	        for root, _, fnames in sorted(os.walk(d)):
	            for fname in sorted(fnames):
	                if fname.endswith('.jpg'):
	                    path = os.path.join(root, fname)
	                    imgs.append(path)

	    rows = [int(image.split('/')[-2]) for image in imgs]
	    row_nums = max(rows) + 1
	    cols = [int(image.split('/')[-1].split('.')[0]) for image in imgs]
	    col_nums = max(cols) +1    
	    
	    # Initialize outputs accordingly:
	    heatmap_res = img_size // stride
	    seg_output = np.zeros((heatmap_res*row_nums, heatmap_res*col_nums,3), dtype=float)

	    seg_model.train(False)
	    seg_model_small.train(False)
	    
	    start_time = time.perf_counter() 

	    for row in tqdm(range(row_nums)):
	        for col in range(col_nums):

	            image_datasets = HeatmapDataset(TILE_DIR, row, col, stride=stride)
	            dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,
	                                                 shuffle=False, num_workers=num_workers)

	            # For Stride 32 (BrainSeg):
	            running_seg = torch.zeros((args.stride), dtype=torch.uint8)
	            output_class = np.zeros((heatmap_res, heatmap_res,3), dtype=float)
	            
	            for idx, data in enumerate(dataloader):
	                # get the inputs
	                inputs = data
	                # wrap them in Variable
	                if use_gpu:
	                    with torch.no_grad():
	                            

	                        inputs = Variable(inputs.cuda(args.gpu))

	                        predict = seg_model(inputs)
	                        _, indices = torch.max(predict.data, 1) # indices = 0:Background, 1:WM, 2:GM
	                        indices = indices.type(torch.uint8)
	                        running_seg =  indices.data.cpu()

	                        softmaxxxed = F.softmax(predict)

	                        softmaxxxed = softmaxxxed.cpu().numpy()
	                        
	                        if isMultiFOV == 1:
		                        if 2 in running_seg[idx_conf]:	                                
	                                # Assuming you have img tensor already defined
	                                batch_size, channels, height, width = inputs.shape

	                                # Top-left corner
	                                img1 = inputs[:, :, 0:height//2, 0:width//2]

	                                # Top-right corner
	                                img2 = inputs[:, :, 0:height//2, width//2:width]

	                                # Bottom-left corner
	                                img3 = inputs[:, :, height//2:height, 0:width//2]

	                                # Bottom-right corner
	                                img4 = inputs[:, :, height//2:height, width//2:width]

	                                listy = [softmaxxxed]
	                                for img in [img1,img2,img3,img4]:
	                                    predict = seg_model_small(img)

			                            _, indices = torch.max(predict.data, 1) # indices = 0:Background, 1:WM, 2:GM  
			                            softmaxxxed_small = F.softmax(predict)

			                            softmaxxxed_small = softmaxxxed_small.cpu().numpy()
			                            
			                            listy.append(softmaxxxed_small)
	                            else:
	                            	listy = [softmaxxxed,softmaxxxed,softmaxxxed,softmaxxxed,softmaxxxed]
		                        
		                        stacked_array = np.stack(tuple(listy), axis=0)

		                        # For Stride 32 (BrainSeg) :
		                        i = (idx // (heatmap_res//batch_size))
		                        j = (idx % (heatmap_res//batch_size))
		                        transposed_array = np.transpose(stacked_array, (1, 2, 0))
		                        output_class[i,j*batch_size:(j+1)*batch_size] = transposed_array

		                    else:
	                        
		                        # For Stride 32 (BrainSeg) :
		                        i = (idx // (heatmap_res//batch_size))
		                        j = (idx % (heatmap_res//batch_size))
		                        output_class[i,j*batch_size:(j+1)*batch_size,:] = softmaxxxed
	            
	            # Final Outputs of Brain Segmentation
	            seg_output[row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = output_class
	            
	    np.save(SAVE_NP_DIR+filename, seg_output)
	    
	    # Time Statistics for Inference
	    end_time = time.perf_counter()
	    print("Time to process " \
	          + filename \
	          + ": ", end_time-start_time, "sec")
	

if __name__ == "__main__":
    main()