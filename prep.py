"""
To pass through model images will have to be grayscale and the correct width and 
height. They will also need to converted into an array. Arrays will need to be 
converted into a torch Tensor.
"""
import cv2
import numpy as np
import torch


def img(ori_img, img_wid, img_hei):
    """Converts image to grayscale and then into an array"""
    img = cv2.imread(ori_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_wid, img_hei))
    img_arr = np.array(img)
    return img, img_arr
    
def arr(img_arr, img_wid, img_hei):
    """Prepares array for pass through model"""
    X = torch.Tensor(img_arr).view(-1, 1, img_wid, img_hei)
    X = X/255.0
    return X




