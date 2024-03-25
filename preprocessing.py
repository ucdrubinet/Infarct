import os
#import os.path
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import pyvips as Vips
from tqdm import tqdm
from utils import vips_utils, normalize
from torchvision import transforms, utils
import time
import torchvision.models as models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image, ImageFile
import statistics
from typing import Optional, Tuple
import pylibczi
from pylibczi import CziScene
import czifile
from czifile import CziFile 
import xml.etree.ElementTree as ET
import argparse
import gc 
import psutil
import resource
import platform
import pickle
import xmltodict
import time
import matplotlib.path as mpath
from skimage.draw import polygon
import cv2
import copy
import shutil

import torch
import random
import torchvision
import torch.nn as nn
from skimage import io, transform
from skimage.morphology import convex_hull_image

import os
import math

import scipy.ndimage
import skimage.io
from skimage.measure import label, regionprops
from skimage.morphology import square
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from sklearn.metrics import confusion_matrix
from skimage.transform import resize

from bs4 import BeautifulSoup

import argparse

def grabCZI(path, verbose = False):
    img = czifile.imread(path)
    if verbose:
        print(img.shape)
        print(img)
    
    img = np.array(img, dtype = np.uint8)
    
    scenes = img.shape[0]
    time = img.shape[1]
    height = img.shape[2]
    width = img.shape[3]
    channels = img.shape[4]
    
    
    img = img.reshape((height, width, channels))
    if verbose:
        print(img)
        print(img.shape) 
        
    dtype_to_format = {
        'uint8': 'uchar',
        'int8': 'char',
        'uint16': 'ushort',
        'int16': 'short',
        'uint32': 'uint',
        'int32': 'int',
        'float32': 'float',
        'float64': 'double',
        'complex64': 'complex',
        'complex128': 'dpcomplex',
    }
    
    ###codes from numpy2vips
    height, width, bands = img.shape
    img = img.reshape(width * height * bands)
    vips = Vips.Image.new_from_memory(img.data, width, height, bands,
                                      dtype_to_format['uint8'])
    try: 
        del img, height, width, bands
        gc.collect()
    except: 
        pass
    
    return vips

def tile(TILE_SIZE,WSI_DIR,SAVE_DIR,CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,imagenames):
	print("Starting tiling....")
	for imagename in tqdm(imagenames[:]):
	    start = time.time()
	    if imagename.split('.')[-1] == 'svs':
	        NAID = imagename.split('.')[0]
	        print("Loading", imagename, " ......")
	        vips_img = Vips.Image.new_from_file(WSI_DIR + imagename, level=0)
	        
	        print("Pre resize: ", vips_img.height, "x", vips_img.width)
	        
	        if NAID in ['NA5009-16_HE', 'NA5009-17_HE', 'NA5009-18_HE']:
	            pass
	        else:
	            print("Resizing small case")
	            vips_img = vips_img.resize(2)
	            
	        print("Post resize: ", vips_img.height, "x", vips_img.width)
	            
	        print("Loaded Image: " + WSI_DIR + imagename)

	        
	        
	        vips_utils.save_and_tile(vips_img, os.path.splitext(imagename)[0], SAVE_DIR, tile_size = TILE_SIZE)
	        print("Done Tiling: ", WSI_DIR + imagename)
	        
	    elif imagename.split('.')[-1] == 'czi':
	        NAID = imagename.split('.')[0]
	        print("Loading", imagename, " ......")
	        try: 
	            vips_img = grabCZI(WSI_DIR + imagename)
	            print("Loaded Image: " + WSI_DIR + imagename)
	            
	            print("Pre resize: ", vips_img.height, "x", vips_img.width)
	            
	        
	            vips_utils.save_and_tile(vips_img, os.path.splitext(imagename)[0], SAVE_DIR, tile_size = TILE_SIZE)

	            print("Done Tiling: ", WSI_DIR + imagename)
	            del vips_img
	            gc.collect()
	            print("Finish Delete", WSI_DIR + imagename)
	        except:
	            print("Error in tiling")
	        
	        

	    else:
	        print("Skipped,", imagename, '. This file is either not .czi or .svs, or not the file assigned')
	    
	    try: 
	        del vips_img 
	        gc.collect()
	    except:
	        pass
	    
	    print("processed in ", time.time()-start," seconds")
	    print("____________________________________________")


def rename_tile(TILE_SIZE,WSI_DIR,SAVE_DIR,CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR):
	print("About to change names to add coordinates")
	for case_folder in sorted(os.listdir(SAVE_DIR)):
	    NAID = case_folder
	    print('Processing NAID: ', NAID)
	    for tile_folder in sorted(os.listdir(SAVE_DIR+NAID+'/0/')):
	        # folder_level == y axis distance determinant
	        # file_level == x axis distance determinant
	        y0 = int(tile_folder)*TILE_SIZE
	        y1 = (int(tile_folder)+1)*TILE_SIZE
	        for tile_file in sorted(os.listdir(SAVE_DIR+NAID+'/0/'+tile_folder+'/')):
	            x0 = int(tile_file.split('.')[0])*TILE_SIZE
	            x1 = (int(tile_file.split('.')[0])+1)*TILE_SIZE
	            os.rename(SAVE_DIR+NAID+'/0/'+str(tile_folder)+'/'+tile_file, 
	                      SAVE_DIR+NAID+'/0/'+str(tile_folder)+'/'+str(y0)+'_'+str(x0)+'_'+str(y1)+'_'+str(x1)+'_'+tile_file)


def extract_annotation(TILE_SIZE,WSI_DIR,SAVE_DIR,CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,imagenames):
	for imagename in tqdm(imagenames[:]):
	    start = time.time()
	    if imagename.split('.')[-1] == 'svs':
	        pass
	        
	    elif imagename.split('.')[-1] == 'czi':
	        NAID = imagename.split('.')[0]
	        print("Loading", imagename, " ......")
	        try: 
	            czifile = pylibczi.CziFile(WSI_DIR + imagename, metafile_out = CZ_DIR + NAID + '.cz',use_pylibczi=True, verbose=True)
	            czifile.read_meta()
	            
	            tree = ET.parse(CZ_DIR + NAID + '.cz') 
	            root = tree.getroot() 
	            
	            tree.write(CZ_DIR + NAID + '.xml')
	        except:
	            print("Error in extracting annotation")
	            
	    print("processed in ", time.time()-start," seconds")
	    print("____________________________________________")


def grab_svs_annot(path):
    doc = xmltodict.parse(open(path, 'r', encoding='utf-8').read())
    
    all_annot = []
    if type(doc['Annotations']['Annotation']['Regions']['Region']) == list:
        for region_idx in range(len(doc['Annotations']['Annotation']['Regions']['Region'])):
            coords = []
            for annot in doc['Annotations']['Annotation']['Regions']['Region'][region_idx]['Vertices']['Vertex']:
                coord = [int(annot['@Y']),int(annot['@X'])]

                coords.append(coord)
            all_annot.append(coords)
    else:
        coords = []
        for annot in doc['Annotations']['Annotation']['Regions']['Region']['Vertices']['Vertex']:
            coord = [int(annot['@Y']),int(annot['@X'])]

            coords.append(coord)
        all_annot.append(coords)
    
    return all_annot

#from: https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-dictionaries-and-lists
def findkeys(node, kv):
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
               yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x

def grab_czi_annot(path):
    doc = xmltodict.parse(open(path, 'r', encoding='utf-8').read())
    
    amt_annot = len(list(findkeys(doc,'Points')))
    
    all_annot = []
    coords = []
    
    #print("AMOUNT ANNOT = ", amt_annot)
    
    for annot_idx in range(amt_annot):
        coords = []
        all_cord = list(findkeys(doc,'Points'))[annot_idx].split(' ')
        isGood = False
        for xy in all_cord:
            #print(xy)
            if int(float(xy.split(',')[1])) < 0 or int(float(xy.split(',')[0])) < 0:
                pass
                #print("Passed on negative coordinates")
            else:
                coord = [int(float(xy.split(',')[1])),int(float(xy.split(',')[0]))]
                isGood = True
                
                coords.append(coord)
        if isGood:
            all_annot.append(coords)
    
    return all_annot



def create_polygon_mask(vertex_coordinates, mask):
    
    rr, cc = polygon(vertex_coordinates[:,0],vertex_coordinates[:,1],mask.shape)
    mask[rr,cc] = 1
    
    return mask


# https://stackoverflow.com/questions/41925853/fill-shapes-contours-using-numpy
def fill_contours(arr):
    return np.maximum.accumulate(arr,1) & \
           np.maximum.accumulate(arr[:,::-1],1)[:,::-1]


def build_maks(TILE_SIZE,WSI_DIR,SAVE_DIR,CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,imagenames):
	gt_df = pd.read_csv('gt.csv')

	for imagename in tqdm(imagenames[:]):
	    start = time.time()

	    wsi_gt = gt_df[gt_df.cases == imagename].iat[0,2]
	    
	    resized = False
	    if wsi_gt == 1:
	        if imagename.split('.')[-1] == 'svs':
	            NAID = imagename.split('.')[0]
	            print("Loading", imagename, " ......")
	            if True: 
	                vips_img = Vips.Image.new_from_file(WSI_DIR + imagename, level=0)
	                print("Loaded Image: " + WSI_DIR + imagename)
	                
	                if NAID in ['NA5009-16_HE', 'NA5009-17_HE', 'NA5009-18_HE']:
	                    pass
	                    #vips_img = vips_img.resize(0.25)
	                else:
	                    print("Resizing small case")
	                    vips_img = vips_img.resize(2)
	                    resized = True

	                dimension = [vips_img.height, vips_img.width]

	                all_annot = grab_svs_annot(CZ_DIR + NAID + '.xml')

	                mask = np.zeros(dimension,'uint8')
	                
	                # It is all matching xml up to here
	                #print(all_annot)
	                
	                print("Starting to build mask")
	                print("Number of annotation groups =", len(all_annot))
	                for coords in all_annot:
	                    #Old method - faulty
	                    #mask = create_polygon_mask(coords, mask)
	                    
	                    #New method
	                    max_y = 0
	                    max_x = 0
	                    min_y = 10000000000
	                    min_x = 10000000000
	                    
	                    submask = np.zeros(dimension,'uint8')
	                    for coord in coords:
	                        y = coord[0]
	                        x = coord[1]
	                        
	                        #print("y = ", y, "x = ", x)
	                        if resized:
	                            y = y*2
	                            x = x*2
	                            submask[y][x] = 1
	                        else:
	                            submask[y][x] = 1
	                            
	                        if y > max_y:
	                            max_y = y
	                            
	                        if x > max_x:
	                            max_x = x
	                            
	                        if y < min_y:
	                            min_y = y
	                            
	                        if x < min_x:
	                            min_x = x
	                    
	                    submask[min_y:max_y+1, min_x:max_x+1] = convex_hull_image(submask[min_y:max_y+1, min_x:max_x+1])
	                    
	                    mask = mask + submask
	                        
	                
	                #print("Filling contours")
	                #mask = fill_contours(mask)

	                #mask = np.packbits(mask,axis=None)
	                #print("Binarized mask")
	                
	                print("Saving mask contours")
	                np.save(MASK_DIR+NAID+'.npy',mask)

	                print("processed in ", time.time()-start," seconds")
	                print("____________________________________________")

	            else:
	                print("Error in generating masks from polygon for ", NAID)


	        elif imagename.split('.')[-1] == 'czi':
	            NAID = imagename.split('.')[0]
	            print("Loading", imagename, " ......")
	            if True: 
	                print(WSI_DIR + imagename)
	                #vips_img = grabCZI(WSI_DIR + imagename)
	                
	                
	                info_path = '/data/Infarct/Infarct_Codes/patched_512/'+str(NAID)+'/vips-properties.xml'
	    
	                print("info path = ", info_path)

	                with open(info_path, 'r') as f:
	                    xml = f.read()

	                soup = BeautifulSoup(xml, "xml")

	                width = soup.find("name", text="width").find_next_sibling("value").text
	                height = soup.find("name", text="height").find_next_sibling("value").text

	                
	                #print("Loaded Image: " + WSI_DIR + imagename)

	                #dimension = [vips_img.height, vips_img.width]
	                dimension = [int(height), int(width)]
	                all_annot = grab_czi_annot(CZ_DIR + NAID + '.xml')

	                mask = np.zeros(dimension,'uint8')
	                
	                # It is all matching up to here
	                #print(all_annot)
	                
	                print("Starting to build mask")
	                print("Number of annotation groups =", len(all_annot))
	                for coords in all_annot:
	                    #Old method - faulty
	                    #mask = create_polygon_mask(coords, mask)
	                    
	                    #New method
	                    max_y = 0
	                    max_x = 0
	                    min_y = 10000000000
	                    min_x = 10000000000
	                    
	                    submask = np.zeros(dimension,'uint8')
	                    for coord in coords:
	                        y = coord[0]
	                        x = coord[1]
	                        
	                        #print("y = ", y, "x = ", x)
	                        if resized:
	                            y = y*2
	                            x = x*2
	                            submask[y][x] = 1
	                        else:
	                            submask[y][x] = 1
	                            
	                        if y > max_y:
	                            max_y = y
	                            
	                        if x > max_x:
	                            max_x = x
	                            
	                        if y < min_y:
	                            min_y = y
	                            
	                        if x < min_x:
	                            min_x = x
	                    
	                    submask[min_y:max_y+1, min_x:max_x+1] = convex_hull_image(submask[min_y:max_y+1, min_x:max_x+1])
	                    
	                    mask = mask + submask
	                        
	                #print("Filling contours")
	                #mask = fill_contours(mask)
	                

	                #mask = np.packbits(mask,axis=None)
	                #print("Binarized mask")
	                
	                print("Saving mask contours")
	                np.save(MASK_DIR+NAID+'.npy',mask) 

	                print("processed in ", time.time()-start," seconds")
	                print("____________________________________________")
	            else:
	                print("Error in generating masks from polygon for ", NAID)
	        
	        
	    else:
	        print("Skipped,", imagename, '. This file is either not .czi or .svs, or not the file assigned')
	    
	    if wsi_gt == 1:
	        try:
	            del vips_img,coords,mask,submask
	            gc.collect()
	        except:
	            pass


def find_BG(TILE_SIZE,WSI_DIR,SAVE_DIR,CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,gpu):
	MODEL_SEG_DIR = 'BrainSec.pt'

	seg_model = torchvision.models.resnet18()
	seg_model.fc = nn.Linear(512, 3)

	checkpoint = torch.load(MODEL_SEG_DIR)
	seg_model.load_state_dict(checkpoint['model_state_dict'])

	device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
	seg_model.to(device)

	for NAID in os.listdir(SAVE_DIR):
	    start = time.time()
	    print("Processing ", NAID)

	    try:
	        os.makedirs(SEGMENTATION_TILE_DIR+NAID)
	    except:
	        pass

	    try:
	        os.makedirs(SEGMENTATION_TILE_DIR+NAID+'/BG/')
	    except:
	        pass

	    for tile_folder in os.listdir(SAVE_DIR+NAID+'/0/'):
	        for tile in os.listdir(SAVE_DIR+NAID+'/0/'+tile_folder):
	            seg_model.train(False)

	            img = Image.open(SAVE_DIR+NAID+'/0/'+tile_folder+'/'+tile)

	            img_tensor = transforms.ToTensor()(img)
	            img_tensor = transforms.Normalize(mean=[0.4409763317567454, 0.4016568471536302, 0.4988298669112181],
                                 std=[0.31297803931100737, 0.2990562933047881, 0.33747493782548915])(img_tensor)
	            img_tensor = TF.resize(img_tensor,512)
	            img_tensor = torch.reshape(img_tensor,(1,3,512,512))
	            img_tensor = img_tensor.cuda(gpu)

	            predict = seg_model(img_tensor)
	            preds = F.sigmoid(predict)
	            _, indices = torch.max(predict.data, 1) # indices = 0:Background, 1:WM, 2:GM
	            indices = indices.type(torch.uint8)
	            running_seg =  indices.data.cpu()


	            if running_seg == 0:# or running_seg2 == 0:
	                shutil.move(SAVE_DIR+NAID+'/0/'+tile_folder+'/'+tile, SEGMENTATION_TILE_DIR+NAID+'/BG/'+tile)
	            else:
	                pass
                    #print("bg removed")

	    print("processed in ", time.time()-start," seconds")
	    print("____________________________________________")


def classify_tiles(TILE_SIZE,WSI_DIR,SAVE_DIR,CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,imagenames):
	gt_df = pd.read_csv('gt.csv')

	for imagename in tqdm(imagenames[:]):
	    start = time.time()

	    wsi_gt = gt_df[gt_df.cases == imagename].iat[0,2]
	    
	    NAID = imagename.split('.')[0]
	    print("Loading", imagename, " ......")

	    try:
	        os.makedirs(SEGMENTATION_TILE_DIR+NAID+'/Heal/')
	    except:
	        pass
	    
	    if wsi_gt == 1:
	        try:
	            os.makedirs(SEGMENTATION_TILE_DIR+NAID+'/Inf/')
	        except:
	            pass
	        mask = np.load(MASK_DIR+NAID+'.npy')
	    
	        print("Loaded mask at ", MASK_DIR+NAID+'.npy' ,", about to process...")

	    #mask = np.unpackbits(mask,count=im_size).reshape(img_dim).view(bool)
	    #mask = mask.view(np.uint8)
	    #print("Unbinarized mask")

	    for tile_folder in os.listdir(SAVE_DIR+NAID+'/0/'):
	        for tile in os.listdir(SAVE_DIR+NAID+'/0/'+tile_folder):
	            
	            try:
	                #file naming convention --> y0 x0 y1 x1
	                y0 = int(tile.split('_')[0])
	                x0 = int(tile.split('_')[1])
	                y1 = int(tile.split('_')[2])
	                x1 = int(tile.split('_')[3])

	                if wsi_gt == 1:
	                    if np.sum(np.sum(mask[y0:y1,x0:x1])) >= (((TILE_SIZE/6)*(TILE_SIZE/6)/2)):
	                        shutil.move(SAVE_DIR+NAID+'/0/'+tile_folder+'/'+tile, SEGMENTATION_TILE_DIR+NAID+'/Inf/'+tile)
	                    elif np.sum(np.sum(mask[y0:y1,x0:x1])) == 0:
	                        shutil.move(SAVE_DIR+NAID+'/0/'+tile_folder+'/'+tile, SEGMENTATION_TILE_DIR+NAID+'/Heal/'+tile)
	                    else:
	                        pass
	                else:
	                    shutil.move(SAVE_DIR+NAID+'/0/'+tile_folder+'/'+tile, SEGMENTATION_TILE_DIR+NAID+'/Heal/'+tile)
	            except:
	                print("Issue with tile ", tile)
	                
	    print("processed in ", time.time()-start," seconds")
	    print("____________________________________________")




def main():
    parser = argparse.ArgumentParser(description="Preprocess parameters.")

    # Define arguments with their default values
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to be used for BG segmentation")
    parser.add_argument("--wsi_dir", type=str, default='../Infarct_dataset/github_test_luca/', help="Directory of Whole Slide Images")
    parser.add_argument("--save_dir", type=str, default='inf_tiles/', help="Directory to save the patches for heatmaps")
    parser.add_argument("--cz_dir", type=str, default='annotation/', help="Directory for CZ annotation files")
    parser.add_argument("--mask_dir", type=str, default='masks/', help="Directory for mask files")
    parser.add_argument("--segmentation_tile_dir", type=str, default='seg_data/', help="Directory for patched images")
    parser.add_argument("--wsi_tile_dir", type=str, default='wsi_level_data/', help="Directory for WSI metadata")

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the arguments as variables in your code
    print(f"Tile Size: {args.tile_size}")
    print(f"Heatmap Tile Size: {args.tile_size*6}")
    print(f"GPU: {args.gpu}")
    print(f"WSI Directory: {args.wsi_dir}")
    print(f"Save Directory: {args.save_dir}")
    print(f"CZ Directory: {args.cz_dir}")
    print(f"Mask Directory: {args.mask_dir}")
    print(f"Segmentation Tile Directory: {args.segmentation_tile_dir}")
    print(f"WSI Tile Directory: {args.wsi_tile_dir}")
                                                                 

    TILE_SIZE = args.tile_size
    gpu = args.gpu

    WSI_DIR = args.wsi_dir

    SAVE_DIR = args.save_dir


    CZ_DIR = args.cz_dir
    MASK_DIR = args.mask_dir

    SEGMENTATION_TILE_DIR = args.segmentation_tile_dir
    WSI_TILE_DIR = args.wsi_tile_dir


    if not os.path.exists(WSI_DIR):
        print("WSI folder does not exist, script should stop now")
    else:
        if not os.path.exists(SEGMENTATION_TILE_DIR):
            print("Tile folder you provided us does not exist, being created now...")
            os.makedirs(SEGMENTATION_TILE_DIR)

        if not os.path.exists(WSI_TILE_DIR):
            print("Tile folder for WSI-level you provided us does not exist, being created now...")
            os.makedirs(WSI_TILE_DIR)

        if not os.path.exists(SAVE_DIR):
            print("Tile folder you provided us does not exist, being created now...")
            os.makedirs(SAVE_DIR)

        if not os.path.exists(CZ_DIR):
            print("Annotation folder you provided us does not exist, being created now...")
            os.makedirs(CZ_DIR)

        if not os.path.exists(MASK_DIR):
            print("Mask folder you provided us does not exist, being created now...")
            os.makedirs(MASK_DIR)

        print("Found WSI folder... ")
        wsi_slides = os.listdir(WSI_DIR)
        imagenames = sorted(wsi_slides)
        print("All WSIs in wsi_dir: ")
        print(imagenames)

    tile(TILE_SIZE*6,WSI_DIR,SAVE_DIR,CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,imagenames)
    print("WSI tiled for heatmaps")
    
    tile(TILE_SIZE,WSI_DIR,'train/',CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,imagenames)
    print("WSI tiled for training")
    
    #shutil.copytree(SAVE_DIR, 'temp/')                                                           
                                                                 
    rename_tile(TILE_SIZE,WSI_DIR,SAVE_DIR,CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR)
    rename_tile(TILE_SIZE,WSI_DIR,'train/',CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR)
    print("Tiles renamed")
    
    extract_annotation(TILE_SIZE,WSI_DIR,'train/',CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,imagenames)
    print("Annotations extracted")
    
    build_maks(TILE_SIZE,WSI_DIR,'train/',CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,imagenames)
    print("Binary masks built")

    find_BG(TILE_SIZE,WSI_DIR,'train/',CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,gpu)
    print("Background tiles labeled")

    classify_tiles(TILE_SIZE,WSI_DIR,'train/',CZ_DIR,MASK_DIR,SEGMENTATION_TILE_DIR,WSI_TILE_DIR,imagenames)
    print("Infarct/Un-infarcted tiles labeled")


if __name__ == "__main__":
    main()