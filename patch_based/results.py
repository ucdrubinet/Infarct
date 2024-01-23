import os
import pickle
import torch
import torchvision
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, Sampler
from torchvision import transforms, utils
from PIL import Image

from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix, accuracy_score, recall_score

import cv2
import numpy as np
import statistics
import torch.nn.functional as F
from bs4 import BeautifulSoup

import math

import scipy.ndimage
import skimage.io
from skimage.measure import label, regionprops
from skimage.morphology import square
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from sklearn.metrics import confusion_matrix
from skimage.transform import resize

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 500

import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt

from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score


def npy_to_image(p,filename, threshold):
    # Load .npy array
    data = np.load(filename)
    
    # Initialize RGB array
    rgb_image = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

    # Loop through each pixel
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j, 0] > data[i, j, 1] and data[i, j, 0] > data[i, j, 2]:
                rgb_image[i, j] = [0, 0, 0]  # Black
            elif data[i, j, 1] > data[i, j, 0] and data[i, j, 1] > data[i, j, 2]:
                if data[i, j, 2] > threshold:
                    rgb_image[i, j] = [0, 255, 255]  # Cyan
                else:
                    rgb_image[i, j] = [255, 255, 0]  # Yellow
            else:
                if data[i, j, 2] > threshold:
                    rgb_image[i, j] = [0, 255, 255]  # Cyan
                else:
                    rgb_image[i, j] = [255, 255, 0]  # Yellow

    # Convert to image and save
    img = Image.fromarray(rgb_image)
    img.save(p)


def check_complete_agreement(data, i, j, threshold):
    # Complete agreement rule: all models must classify as class 2 above threshold
    if data[i, j, 2, 0] >= threshold:
        return all(data[i, j, 2, model_idx] == np.max(data[i, j, :, model_idx]) for model_idx in range(1, 5))
    return False

def check_n_model_agreement(data, i, j, threshold, n):
    # N models agreement rule: at least 'n' models must classify as class 2 above threshold
    count_agreement = sum(data[i, j, 2, model_idx] == np.max(data[i, j, :, model_idx]) for model_idx in range(5))
    return count_agreement >= n and data[i, j, 2, 0] >= threshold

def check_any_model_confidence(data, i, j, conf_threshold):
    # Any model confidence rule: any of the following four softmaxes have class 2 confidence above conf_threshold
    return any(data[i, j, 2, model_idx] >= conf_threshold for model_idx in range(1, 5))
    
def npy_to_multi_model_image(p,filename, threshold, cyan_rule='complete_agreement',n=2,conf_threshold=0.9):
    # Load .npy array
    data = np.load(filename)
    
    # Initialize RGB array
    rgb_image = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

    # Loop through each pixel
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Identify the class with the highest confidence from the first softmax
            first_model_max_class = np.argmax(data[i, j, :, 0])
            
            # Check cyan classification based on the cyan_rule
            is_cyan = False
            if cyan_rule == 'complete_agreement':
                is_cyan = check_complete_agreement(data, i, j, threshold)
            elif cyan_rule == 'n_agreement':
                is_cyan = check_n_model_agreement(data, i, j, threshold, n)
            elif cyan_rule == 'confidence_threshold':
                is_cyan = check_any_model_confidence(data, i, j, conf_threshold)
            else:
                print("Name chosen for MultiFOV committee rule was incorrect, choosing complete_agreement")
                is_cyan = check_complete_agreement(data, i, j, threshold)
            
            # Apply color based on classification
            if first_model_max_class == 0:
                rgb_image[i, j] = [0, 0, 0]  # Black
            elif first_model_max_class == 1:
                rgb_image[i, j] = [255, 255, 0]  # Yellow
            elif first_model_max_class == 2 and is_cyan:
                rgb_image[i, j] = [0, 255, 255]  # Cyan
            else:
                rgb_image[i, j] = [255, 255, 0]  # Yellow

    # Convert to image and save
    img = Image.fromarray(rgb_image)
    img.save(p)


def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def calculate_map(ground_truth_bboxes, prediction_bboxes, iou_thresh):
    # Handling the case where there are no ground truth bboxes and no predictions
    if not ground_truth_bboxes and not prediction_bboxes:
        return 1.0, 0, 0, 0

    # Handling the case where there are no ground truth bboxes but there are predictions
    if not ground_truth_bboxes:
        return 0.0, 0, len(prediction_bboxes), 0

    matches = []
    
    # For each predicted bbox, find if it matches with a ground truth bbox
    for pred_bbox in prediction_bboxes:
        match_found = False
        for gt_bbox in ground_truth_bboxes:
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou >= iou_thresh:
                matches.append((pred_bbox, gt_bbox, iou))
                match_found = True
                break  # Break after the first match
        if not match_found:
            matches.append((pred_bbox, None, 0))  # No match found, false positive

    # Sort by IoU in descending order for matched bboxes
    matches.sort(key=lambda x: x[2], reverse=True)

    tp, fp = 0, 0
    precision, recall = [], []
    matched_gt_bboxes = set()

    for i, match in enumerate(matches):
        if match[1] is not None and match[1] not in matched_gt_bboxes:
            tp += 1
            matched_gt_bboxes.add(match[1])
        else:
            fp += 1
        
        precision.append(tp / (tp + fp))
        recall.append(tp / len(ground_truth_bboxes))

    # Calculate Average Precision
    ap = sum(precision) / len(precision) if precision else 0

    return ap, tp, fp, len(ground_truth_bboxes)



def calculate_map_large(ground_truth_bboxes, prediction_bboxes, iou_thresh,size_thresh):
    
    gtb = []
    for gtbbox in ground_truth_bboxes:
        area = (gtbbox[2]-gtbbox[0])*(gtbbox[3]-gtbbox[1])
        
        if area > size_thresh:
            gtb.append(gtbbox)
    
    ground_truth_bboxes = gtb
            
    pdb = []
    for ptdbox in prediction_bboxes:
        area = (ptdbox[2]-ptdbox[0])*(ptdbox[3]-ptdbox[1])
        
        if area > size_thresh:
            pdb.append(ptdbox)
        
    prediction_bboxes = pdb
    
    
    
    # Handling the case where there are no ground truth bboxes and no predictions
    if not ground_truth_bboxes and not prediction_bboxes:
        return 1.0, 0, 0, 0

    # Handling the case where there are no ground truth bboxes but there are predictions
    if not ground_truth_bboxes:
        return 0.0, 0, len(prediction_bboxes), 0

    matches = []
    
    # For each predicted bbox, find if it matches with a ground truth bbox
    for pred_bbox in prediction_bboxes:
        match_found = False
        for gt_bbox in ground_truth_bboxes:
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou >= iou_thresh:
                matches.append((pred_bbox, gt_bbox, iou))
                match_found = True
                break  # Break after the first match
        if not match_found:
            matches.append((pred_bbox, None, 0))  # No match found, false positive

    # Sort by IoU in descending order for matched bboxes
    matches.sort(key=lambda x: x[2], reverse=True)

    tp, fp = 0, 0
    precision, recall = [], []
    matched_gt_bboxes = set()

    for i, match in enumerate(matches):
        if match[1] is not None and match[1] not in matched_gt_bboxes:
            tp += 1
            matched_gt_bboxes.add(match[1])
        else:
            fp += 1
        
        precision.append(tp / (tp + fp))
        recall.append(tp / len(ground_truth_bboxes))

    # Calculate Average Precision
    ap = sum(precision) / len(precision) if precision else 0
    
#     print("tp = ", tp)
#     print("fp = ", fp)

    return ap, tp, fp, len(ground_truth_bboxes)


def find_connected_components(image,area_thresh=50):
    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list of bounding rectangles for the contours
    bounding_rects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Check if the connected component has white pixels more than a threshold
        #if area >= 50:
        if area >= int(area_thresh):
            # Get the bounding box of the connected component (x, y, width, height)
            x, y, w, h = cv2.boundingRect(contour)
            bounding_rects.append((x, y, x + w, y + h))

    return bounding_rects

def calculate_distance(rect1, rect2):
    # Calculate the distance between two rectangles based on their centroids
    centroid1 = ((rect1[0] + rect1[2]) // 2, (rect1[1] + rect1[3]) // 2)
    centroid2 = ((rect2[0] + rect2[2]) // 2, (rect2[1] + rect2[3]) // 2)
    return np.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)

def group_connected_components(rectangles, distance_threshold):
    # Group connected components that are close to each other
    groups = []
    for rect in rectangles:
        is_grouped = False
        for group in groups:
            for g_rect in group:
                if calculate_distance(rect, g_rect) < distance_threshold:
                    group.append(rect)
                    is_grouped = True
                    break
            if is_grouped:
                break
        if not is_grouped:
            groups.append([rect])

    return groups

def draw_bounding_box(image, rect):
    # Draw bounding box on the image
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)  # Green color, thickness 2

    return image

def find_box_coordinates(image,threshold_area = 20, distance_threshold=150,area_thresh=50,kernelsz=8):
    #gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    gray = image
    
    # Threshold the grayscale image to obtain a binary mask
    _, binary_image = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    rects=[]

    distance_threshold = 150  # Distance threshold for grouping connected components
    distance_threshold = int(distance_threshold)
    
    #print(binary_image)
    
    prediction_image_og = binary_image
    
    prediction_image = binary_image
    
    prediction_image = scipy.ndimage.morphology.binary_erosion(prediction_image)
    #Dilation
    prediction_image = scipy.ndimage.morphology.binary_dilation(prediction_image)

    # Perform connected component analysis
    labelled_mask = label(prediction_image)
    regions = regionprops(labelled_mask)

    # Define a threshold based on region properties (e.g., area)
    threshold_area = 20  # Adjust as needed
    threshold_area = int(threshold_area)

    # Create a new mask with only the regions above the threshold area
    prediction_image = np.zeros_like(prediction_image_og)
    for region in regions:
        if region.area > threshold_area:
            prediction_image[labelled_mask == region.label] = 1


    kernel_size_dil = 8
    kernel_size_dil = int(kernelsz)
    
    structuring_element = square(kernel_size_dil)

    # Perform dilation on the filtered_mask using the structuring element

    prediction_image = scipy.ndimage.morphology.binary_dilation(prediction_image, structure=structuring_element)
    
    binary_image = prediction_image
    
    #print(binary_image)
    
    binary_image = binary_image.astype(np.uint8)
    
    #print(binary_image)
    
    # Find connected components with white pixels
    bounding_rects = find_connected_components(binary_image,area_thresh=area_thresh)
    
    #print("bouding rects = ", bounding_rects)

    # Group connected components that are close to each other
    groups = group_connected_components(bounding_rects, distance_threshold)
    
    #print("groups = ", groups)

    # Draw bounding boxes for each group on the image
    image_with_boxes = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for group in groups:
    # Create a bounding box that covers all connected components in the group
        x_min = min(rect[0] for rect in group)
        y_min = min(rect[1] for rect in group)
        x_max = max(rect[2] for rect in group)
        y_max = max(rect[3] for rect in group)
        combined_rect = (x_min, y_min, x_max, y_max)
        #print(combined_rect)
        rects.append(combined_rect)
        # Draw the bounding box on the image
        #image_with_boxes = draw_bounding_box(image_with_boxes, combined_rect)
    
    rects = remove_contained_boxes(rects)
    
    for combined_rect in rects:
        image_with_boxes = draw_bounding_box(image_with_boxes, combined_rect)
    
    plt.imshow(cv2.cvtColor(prediction_image_og, cv2.COLOR_GRAY2BGR), interpolation='nearest')
    plt.show()
    plt.close()
    
    plt.imshow(image_with_boxes, interpolation='nearest')
    plt.show()
    plt.close()
    
    #output_path = 'NA5089_pred_erod_dil8_Postprocess_newp.png'  # Replace with the desired output path
    #cv2.imwrite(output_path, image_with_boxes)
    
    
    return rects

def create_mask_from_heatmap(heatmap):
    # Create an empty mask of the same size as heatmap but in grayscale
    mask = np.zeros_like(heatmap[..., 0])
    
    # Set mask pixels to white where heatmap is cyan and to black otherwise
    mask[np.where((heatmap[..., 0] == 255) & (heatmap[..., 1] == 255) & (heatmap[..., 2] == 0))] = 255
    
    return mask


def convert_to_cyan_white_black_grayscale_cv2(image_path):
    # Load the image using cv2
    image = cv2.imread(image_path)

    # Define the RGB range for cyan (Note: OpenCV uses BGR format)
    cyan_min = np.array([255, 255, 0])
    cyan_max = np.array([255, 255, 0])

    # Create a mask identifying where the cyan is in the image
    cyan_mask = np.all(image == cyan_min, axis=-1)

    # Create a new array filled with black
    new_image_data = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Apply the mask to set cyan locations to white
    new_image_data[cyan_mask] = 255

    return new_image_data


def find_case_values(case_name, dataframe):
    # Append possible extensions to the case_name
    possible_cases = [case_name + ext for ext in ['.svs', '.czi']]
    
    # Find the row with the matching case
    matching_row = dataframe[dataframe['cases'].isin(possible_cases)]
    
    # If a match is found, return the gt and WMR values
    if not matching_row.empty:
        return matching_row[['gt', 'WMR']].iloc[0].to_dict()
    else:
        return "No matching case found"
    
def count_cyan_pixels(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the RGB values for cyan
    cyan = np.array([0, 255, 255], dtype="uint8")

    # Find the pixels that match the cyan color
    # Note: The tolerances can be adjusted if an exact match is not required
    mask = cv2.inRange(image_rgb, cyan, cyan)

    # Count the cyan pixels
    count = np.count_nonzero(mask)

    return count

def is_contained(boxA, boxB):
    # Check if boxA is contained in boxB
    return boxA[0] >= boxB[0] and boxA[1] >= boxB[1] and boxA[2] <= boxB[2] and boxA[3] <= boxB[3]

def remove_contained_boxes(prediction_bboxes):
    filtered_bboxes = []

    for i, boxA in enumerate(prediction_bboxes):
        contained = False
        for j, boxB in enumerate(prediction_bboxes):
            if i != j and is_contained(boxA, boxB):
                contained = True
                break

        if not contained:
            filtered_bboxes.append(boxA)

    return filtered_bboxes


def run(MASK_DIR,WSI_DIR,HEATMAP_DIR,gtdf,scale_parameters,distance_threshold,area_thresh_par,threshold_area,kernelsz,predconf,rule,isMultiFOV,micrometer_diameter,n_rule,conf_rule):
    tp8_total = 0 
    fp8_total = 0 

    tp7_total = 0 
    fp7_total = 0 
    
    tp6_total = 0 
    fp6_total = 0 

    tp5_total = 0
    fp5_total = 0
    
    tp4_total = 0
    fp4_total = 0
    
    tp3_total = 0
    fp3_total = 0
    
    tp2_total = 0
    fp2_total = 0
    
    tp1_total = 0
    fp1_total = 0

    len_totalL = 0
    len_n_total = 0

    tp8_totalL = 0 
    fp8_totalL = 0 

    tp7_totalL = 0 
    fp7_totalL = 0 
    
    tp6_totalL = 0 
    fp6_totalL = 0 

    tp5_totalL = 0
    fp5_totalL = 0
    
    tp4_totalL = 0
    fp4_totalL = 0
    
    tp3_totalL = 0
    fp3_totalL = 0
    
    tp2_totalL = 0
    fp2_totalL = 0
    
    tp1_totalL = 0
    fp1_totalL = 0

    WSI_pred = []
    WSI_gt = []

    WSI_pred_strict = []
    WSI_gt_strict = []
    
    WSI_pred_vstrict = []
    WSI_gt_vstrict = []

    WMR_0_1 = []
    WMR_2_3 = []

    for numpy_map in os.listdir(NP_MAP_DIR):

        if isMultiFOV:
            npy_to_multi_model_image(HEATMAP_DIR+numpy_map.split('.')[0]+'.png',NP_MAP_DIR+numpy_map,predconf,cyan_rule=rule,n=n_rule,conf_threshold=conf_rule)
        else:
            npy_to_image(HEATMAP_DIR+numpy_map.split('.')[0]+'.png',NP_MAP_DIR+numpy_map,predconf)
    
    for heatmap_pt in sorted(os.listdir(HEATMAP_DIR)):
        caseID = heatmap_pt.split('.')[0]

        dictr = find_case_values(caseID,gtdf)

        wsi_gt = dictr['gt']
        wsi_wmr = dictr['WMR']

        WSI_gt.append(wsi_gt)

        #print("Processing ", caseID, "with gt ", wsi_gt, " and WMR ", wsi_wmr)

        heatmap_path = HEATMAP_DIR+heatmap_pt

        #print(heatmap_path)

        info_path = WSI_DIR+str(caseID)+'/vips-properties.xml'

        #print("info path = ", info_path)

        with open(info_path, 'r') as f:
            xml = f.read()

        soup = BeautifulSoup(xml, "xml")

        width = soup.find("name", text="width").find_next_sibling("value").text
        height = soup.find("name", text="height").find_next_sibling("value").text

        #print("WSI Dimensions")
        #print(width, "x", height)
        og_area = int(width)*int(height)

        height = int((int(height)/int(width))*1024)
        width = 1024

        #print("Heatmap Dimensions")
        #print(width, "x", height)
        new_area = width*height

        #print("new area / og area ratio = ", new_area/og_area)

        #Half of lacunar size
        #micrometer_diameter = 5000

        area_thresh = ((micrometer_diameter / 2)**2)*(3.14)
        area_large_inf = area_thresh * (new_area/og_area)

        if scale_parameters:
            distance_threshold = distance_threshold * (1/0.0001395055937405748) * (new_area/og_area)

            area_thresh_par = area_thresh_par * (1/0.0001395055937405748) * (new_area/og_area)

            threshold_area = threshold_area * (1/0.0001395055937405748) * (new_area/og_area)

            kernelsz = kernelsz * (1/0.0001395055937405748) * (new_area/og_area)


        inf_count = count_cyan_pixels(heatmap_path)

        if wsi_wmr in [0, 1]:
            WMR_0_1.append(inf_count)
        elif wsi_wmr in [2, 3]:
            WMR_2_3.append(inf_count)

        heatmap = convert_to_cyan_white_black_grayscale_cv2(heatmap_path)

        prediction_image = cv2.resize(heatmap, (width, height))

        prediction_image_og = prediction_image

        predicted_boxes = find_box_coordinates(prediction_image,threshold_area=threshold_area,distance_threshold=distance_threshold,area_thresh=area_thresh_par,kernelsz=kernelsz)

        #print("predicted_boxes = ", predicted_boxes)

        max_area_pbox = 0
        for pbox in predicted_boxes:
            area = (pbox[2]-pbox[0])*(pbox[3]-pbox[1])

            if area > max_area_pbox:
                max_area_pbox = area

        if max_area_pbox > area_large_inf:
            WSI_pred.append(1)
        else:
            WSI_pred.append(0)

        # Get gt path
        ground_truth_image_path =  MASK_DIR+caseID+'.png'

        # Read the ground truth images
        ground_truth_image = cv2.imread(ground_truth_image_path,cv2.IMREAD_GRAYSCALE)

        try:
            ground_truth_image = cv2.resize(ground_truth_image, (width, height))

            # Find bounding box coordinates from the ground truth image
            ground_truth_boxes = find_box_coordinates(ground_truth_image,threshold_area=threshold_area,distance_threshold=distance_threshold,area_thresh=area_thresh_par,kernelsz=kernelsz)
        except:
            ground_truth_boxes = []
        
        
        _, tp8, fp8, len_n = calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.8)
        _, tp7, fp7, len_n = calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.7)
        _, tp6, fp6, len_n = calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.6)
        _, tp5, fp5, len_n = calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.5)
        _, tp4, fp4, len_n = calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.4)
        _, tp3, fp3, len_n = calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.3)
        _, tp2, fp2, len_n = calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.2)
        _, tp1, fp1, len_n = calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.1)
        
        
        _, tp8l, fp8l, len_h = calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.8,size_thresh = area_large_inf)
        _, tp7l, fp7l, len_h = calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.7,size_thresh = area_large_inf)
        _, tp6l, fp6l, len_h = calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.6,size_thresh = area_large_inf)
        _, tp5l, fp5l, len_h = calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.5,size_thresh = area_large_inf)
        _, tp4l, fp4l, len_h = calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.4,size_thresh = area_large_inf)
        _, tp3l, fp3l, len_h = calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.3,size_thresh = area_large_inf)
        _, tp2l, fp2l, len_h = calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.2,size_thresh = area_large_inf)
        _, tp1l, fp1l, len_h = calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.1,size_thresh = area_large_inf)

        if tp1l > 0 or fp1l > 0:
            WSI_pred_strict.append(1)
        else:
            WSI_pred_strict.append(0)

        if len_h > 0:
            WSI_gt_strict.append(1)
        else:
            WSI_gt_strict.append(0)
            
            
        if tp1 > 0 or fp1 > 0:
            WSI_pred_vstrict.append(1)
        else:
            WSI_pred_vstrict.append(0)

        WSI_gt_vstrict.append(wsi_gt)

        tp8_total += tp8
        fp8_total += fp8
        tp7_total += tp7
        fp7_total += fp7 
        tp6_total += tp6
        fp6_total += fp6
        tp5_total += tp5
        fp5_total += fp5
        tp4_total += tp4
        fp4_total += fp4
        tp3_total += tp3
        fp3_total += fp3
        tp2_total += tp2
        fp2_total += fp2
        tp1_total += tp1
        fp1_total += fp1
        len_n_total += len_n

        tp8_totalL += tp8l
        fp8_totalL += fp8l
        tp7_totalL += tp7l
        fp7_totalL += fp7l
        tp6_totalL += tp6l
        fp6_totalL += fp6l
        tp5_totalL += tp5l
        fp5_totalL += fp5l
        tp4_totalL += tp4l
        fp4_totalL += fp4l
        tp3_totalL += tp3l
        fp3_totalL += fp3l
        tp2_totalL += tp2l
        fp2_totalL += fp2l
        tp1_totalL += tp1l
        fp1_totalL += fp1l
        len_totalL += len_h

        print("All predictions")
        
        
        print("mAP IoU > .8 = ", calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.8)[0])
        print("mAP IoU > .7 = ", calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.7)[0])
        print("mAP IoU > .6 = ", calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.6)[0])
        print("mAP IoU > .5 = ", calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.5)[0])
        print("mAP IoU > .4 = ", calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.4)[0])
        print("mAP IoU > .3 = ", calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.3)[0])
        print("mAP IoU > .2 = ", calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.2)[0])
        print("mAP IoU > .1 = ", calculate_map(ground_truth_boxes, predicted_boxes, iou_thresh=0.1)[0])

        print("Large predictions only")        
        print("mAP IoU > .8 = ", calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.8,size_thresh = area_large_inf)[0])
        print("mAP IoU > .7 = ", calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.7,size_thresh = area_large_inf)[0])
        print("mAP IoU > .6 = ", calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.6,size_thresh = area_large_inf)[0])
        print("mAP IoU > .5 = ", calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.5,size_thresh = area_large_inf)[0])
        print("mAP IoU > .4 = ", calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.4,size_thresh = area_large_inf)[0])
        print("mAP IoU > .3 = ", calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.3,size_thresh = area_large_inf)[0])
        print("mAP IoU > .2 = ", calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.2,size_thresh = area_large_inf)[0])
        print("mAP IoU > .1 = ", calculate_map_large(ground_truth_boxes, predicted_boxes, iou_thresh=0.1,size_thresh = area_large_inf)[0])
        print("\n")
        print("\n")

    print("\n")
    print("All predictions")
    list1 = [(tp8_total / (tp8_total + fp8_total)),(tp7_total / (tp7_total + fp7_total)),(tp6_total / (tp6_total + fp6_total)),
         (tp5_total / (tp5_total + fp5_total)), (tp4_total / (tp4_total + fp4_total)), (tp3_total / (tp3_total + fp3_total)),
        (tp2_total / (tp2_total + fp2_total)), (tp1_total / (tp1_total + fp1_total))]
    list1.reverse()
    print(list1)
        
    print("Total mAP IoU > .8 = ", (tp8_total / (tp8_total + fp8_total)))
    print("Total mAP IoU > .7 = ", (tp7_total / (tp7_total + fp7_total)))
    print("Total mAP IoU > .6 = ", (tp6_total / (tp6_total + fp6_total)))
    print("Total mAP IoU > .5 = ", (tp5_total / (tp5_total + fp5_total)))
    print("Total mAP IoU > .4 = ", (tp4_total / (tp4_total + fp4_total)))
    print("Total mAP IoU > .3 = ", (tp3_total / (tp3_total + fp3_total)))
    print("Total mAP IoU > .2 = ", (tp2_total / (tp2_total + fp2_total)))
    print("Total mAP IoU > .1 = ", (tp1_total / (tp1_total + fp1_total)))
    
    list2 = [(tp8_total / (len_n_total)), (tp7_total / (len_n_total)), (tp6_total / (len_n_total)), (tp5_total / (len_n_total)),
          (tp4_total / (len_n_total)), (tp3_total / (len_n_total)), (tp2_total / (len_n_total)), (tp1_total / (len_n_total))]
    list2.reverse()
    print(list2)
    
    print("Total mAR IoU > .8 = ", (tp8_total / (len_n_total)))
    print("Total mAR IoU > .7 = ", (tp7_total / (len_n_total)))
    print("Total mAR IoU > .6 = ", (tp6_total / (len_n_total)))
    print("Total mAR IoU > .5 = ", (tp5_total / (len_n_total)))
    print("Total mAR IoU > .4 = ", (tp4_total / (len_n_total)))
    print("Total mAR IoU > .3 = ", (tp3_total / (len_n_total)))
    print("Total mAR IoU > .2 = ", (tp2_total / (len_n_total)))
    print("Total mAR IoU > .1 = ", (tp1_total / (len_n_total)))

    print("\n")
    print("Large predictions only")
    print([(tp1_totalL / (tp1_totalL + fp1_totalL)), (tp2_totalL / (tp2_totalL + fp2_totalL)), (tp3_totalL / (tp3_totalL + fp3_totalL)),
          (tp4_totalL / (tp4_totalL + fp4_totalL)), (tp5_totalL / (tp5_totalL + fp5_totalL)), (tp6_totalL / (tp6_totalL + fp6_totalL)),
          (tp7_totalL / (tp7_totalL + fp7_totalL)), (tp8_totalL / (tp8_totalL + fp8_totalL))])
    print("Total mAP IoU > .8 = ", (tp8_totalL / (tp8_totalL + fp8_totalL)))
    print("Total mAP IoU > .7 = ", (tp7_totalL / (tp7_totalL + fp7_totalL)))
    print("Total mAP IoU > .6 = ", (tp6_totalL / (tp6_totalL + fp6_totalL)))
    print("Total mAP IoU > .5 = ", (tp5_totalL / (tp5_totalL + fp5_totalL)))
    print("Total mAP IoU > .4 = ", (tp4_totalL / (tp4_totalL + fp4_totalL)))
    print("Total mAP IoU > .3 = ", (tp3_totalL / (tp3_totalL + fp3_totalL)))
    print("Total mAP IoU > .2 = ", (tp2_totalL / (tp2_totalL + fp2_totalL)))
    print("Total mAP IoU > .1 = ", (tp1_totalL / (tp1_totalL + fp1_totalL)))
    
    print([(tp1_totalL / (len_totalL)),(tp2_totalL / (len_totalL)), (tp3_totalL / (len_totalL)),(tp4_totalL / (len_totalL)),
          (tp5_totalL / (len_totalL)), (tp6_totalL / (len_totalL)), (tp7_totalL / (len_totalL)), (tp8_totalL / (len_totalL))])
    print("Total mAR IoU > .8 = ", (tp8_totalL / (len_totalL)))
    print("Total mAR IoU > .7 = ", (tp7_totalL / (len_totalL)))
    print("Total mAR IoU > .6 = ", (tp6_totalL / (len_totalL)))
    print("Total mAR IoU > .5 = ", (tp5_totalL / (len_totalL)))
    print("Total mAR IoU > .4 = ", (tp4_totalL / (len_totalL)))
    print("Total mAR IoU > .3 = ", (tp3_totalL / (len_totalL)))
    print("Total mAR IoU > .2 = ", (tp2_totalL / (len_totalL)))
    print("Total mAR IoU > .1 = ", (tp1_totalL / (len_totalL)))
    print("\n")

    try:
        print("Average Infarct Count WMR (0-1) = ", statistics.mean(WMR_0_1), "+-", statistics.stdev(WMR_0_1))
    except:
        print("Not enough samples in WMR (0-1), here is list -->", WMR_0_1)
    try:
        print("Average Infarct Count WMR (2-3) = ", statistics.mean(WMR_2_3), "+-", statistics.stdev(WMR_2_3))
    except:
        print("Not enough samples in WMR (2-3), here is list -->", WMR_2_3)

    print("\n")
    print("WSI Level Metrics STRICT")
    print(confusion_matrix(WSI_gt_strict,WSI_pred_strict))
    print(classification_report(WSI_gt_strict,WSI_pred_strict))
    print("Acc = ", accuracy_score(WSI_gt_strict,WSI_pred_strict))
    print("Spec = ", recall_score(WSI_gt_strict,WSI_pred_strict,pos_label=0))
    print("Sens = ", recall_score(WSI_gt_strict,WSI_pred_strict,pos_label=1))
    
    print("\n")
    print("WSI Level Metrics VERY STRICT")
    print(confusion_matrix(WSI_gt_vstrict,WSI_pred_vstrict))
    print(classification_report(WSI_gt_vstrict,WSI_pred_vstrict))
    print("Acc = ", accuracy_score(WSI_gt_vstrict,WSI_pred_vstrict))
    print("Spec = ", recall_score(WSI_gt_vstrict,WSI_pred_vstrict,pos_label=0))
    print("Sens = ", recall_score(WSI_gt_vstrict,WSI_pred_vstrict,pos_label=1))

    print("\n")
    print("WSI Level Metrics")
    print(confusion_matrix(WSI_gt,WSI_pred))
    print(classification_report(WSI_gt,WSI_pred))
    
    print("Acc = ", accuracy_score(WSI_gt,WSI_pred))
    print("Spec = ", recall_score(WSI_gt,WSI_pred,pos_label=0))
    print("Sens = ", recall_score(WSI_gt,WSI_pred,pos_label=1))

    return accuracy_score(WSI_gt,WSI_pred), accuracy_score(WSI_gt_strict,WSI_pred_strict)


def main():
    parser = argparse.ArgumentParser(description="Preprocess parameters.")

    # Define arguments with their default values
    parser.add_argument("--heatmap_dir", type=str, default='./Infseg/', help="Directory for heatmap .npy files")
    parser.add_argument("--scale_parameters", type=bool, default=False, help="To use micrometers instead of pixel for parameters")
    parser.add_argument("--dist_threshold", type=int, default=50, help="Distance threshold for grouping connected components")
    parser.add_argument("--area_threshold", type=int, default=50, help="Area threshold for filtering small predictions")
    parser.add_argument("--pred_detection_threshold", type=int, default=5, help="Area threshold for bounding box placement in postprocessing")
    parser.add_argument("--kernel_size", type=int, default=4, help="Kernel size for morphological operations")
    parser.add_argument("--pred_conf", type=float, default=0.95, help="Prediction confidence threshold for patchwise prediction")
    parser.add_argument("--com_rule", type=str, default='complete_agreement', help="Commitee rule for MultiFOV patchwise prediction. Options: complete_agreement, n_agreement, and confidence_threshold")
    parser.add_argument("--n_rule", type=int, default=2, help="N value for committee rule")
    parser.add_argument("--conf_rule", type=float, default=0.9, help="Prediction confidence value for committee rule")

    parser.add_argument("--multiFOV", type=int, default=0, help="0 for single FOV, 1 for multiFOV")
    parser.add_argument("--WSI_level_threshold", type=int, default=5000, help="Threshold (in micrometers) for detection to be turned into WSI level label")
    parser.add_argument("--wsi_dir", type=str, default='../Infarct_dataset/val/', help="Directory of Whole Slide Images")
    parser.add_argument("--mask_dir", type=str, default='masks/', help="Directory for mask files")


    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the arguments as variables in your code
    print(f"Heatmap Directory: {args.heatmap_dir}")
    print(f"Will scale parameters to micrometers: {args.scale_parameters}")
    print(f"Distance threshold for grouping connected components: {args.dist_threshold}")
    print(f"Area Threshold: {args.area_threshold}")
    print(f"Prediction threshold area for bounding box: {args.pred_detection_threshold}")
    print(f"Kernel size for morphological operations: {args.kernel_size}")
    print(f"Prediction confidence: {args.pred_conf}")
    if args.multiFOV == 1:
        print("MultiFOV selected")
        print(f"Committee rule: {args.com_rule}")
        print(f"N value for committee rule: {args.n_rule}")
        print(f"Prediction confidence value for committee rule: {args.conf_rule}")
    print(f"Threshold for WSI level prediction: {args.WSI_level_threshold}")
    print(f"WSI tiled directory: {args.wsi_dir}")
    print(f"Binary mask directory: {args.mask_dir}")



    HEATMAP_DIR = args.heatmap_dir
    scale_parameters = args.scale_parameters
    distance_threshold = args.dist_threshold
    area_thresh_par = args.area_threshold
    threshold_area = args.pred_detection_threshold
    kernelsz = args.kernel_size
    predconf = args.pred_conf
    rule = args.com_rule
    n_rule = args.n_rule
    conf_rule = args.conf_rule
    isMultiFOV = args.multiFOV
    micrometer_diameter = args.WSI_level_threshold
    WSI_DIR = args.wsi_dir
    MASK_DIR = args.mask_dir

    gtdf = pd.read_csv('gt_plus_wmr_pvs.csv')
    run(MASK_DIR,WSI_DIR,HEATMAP_DIR,gtdf,scale_parameters,distance_threshold,area_thresh_par,threshold_area,kernelsz,predconf,rule,isMultiFOV,micrometer_diameter,n_rule,conf_rule)


if __name__ == "__main__":
    main()