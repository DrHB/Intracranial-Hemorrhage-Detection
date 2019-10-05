import PIL
import pydicom
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import fastai
from fastai.core import parallel
import os
import argparse
from functools import partial
import cv2

def window_image(img, window_center, window_width, intercept, slope):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    return img


def get_first_of_dicom_field_as_int(x):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    dicom_fields = [
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def return_single_wd(dcm_in,  wc, ww):
    dcm = pydicom.dcmread(str(dcm_in))
    intercept, slope = get_windowing(dcm)
    img = pydicom.read_file(str(dcm_in)).pixel_array
    img = window_image(img, wc, ww, intercept, slope)
    return img


def crop_image_only_outside(img,tol=0):
    '''
    https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934
    '''
    
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()-1
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()-1
    return img[row_start:row_end,col_start:col_end]


def normalized_data(x):
    return (x-np.min(x))/((np.max(x)-np.min(x))+1e-8)*255 

def convert_to_png(dcm_in, _, dir_img):
    
    #returning each channel as a slice 
    img = [return_single_wd(dcm_in, wc=40, ww=80), 
            return_single_wd(dcm_in, wc=50, ww=175), 
            return_single_wd(dcm_in, wc=500,ww=3000)]
      
    


    #normalizing each channel (between 0 - 1 *255)
    #the reason I am doing this becasue they are some negative values
    
    img = list(map(normalized_data, img))

    try:
        k = np.zeros((512, 512, 3))
        k[:,:, 0] = img[0] #192
        k[:,:, 1] = img[1] #128
        k[:,:, 2] = img[2] #64
    except:
        #in case shape is diffrent than 512, 512 
        w, h = img[0].shape
        k = np.zeros((w, h, 3))
        k[:,:, 0] = img[0] #192
        k[:,:, 1] = img[1] #128
        k[:,:, 2] = img[2] #64
    
    #croping image on outside to remove black background and returning PIL
    img = PIL.Image.fromarray(k.astype('uint8'))
    img.save(os.path.join(dir_img, os.path.basename(dcm_in)[:-3] + 'png'), quality=95)

    

    

def main():
    parser = argparse.ArgumentParser()
    #will probmot help instead of erro message 


    parser.add_argument("-dicom_dir", 
                        help = 'direcotry of the dicom files', 
                        type=str)
    
    parser.add_argument("-output_dir", 
                        help = 'output file name where all the .png files will be stored', 
                        type=str)

    parser.add_argument("-num_workers", 
                        help = 'number_of_cpu', 
                        type=int, 
                        default=16)
    
    
    args = parser.parse_args()
    
    try:
        os.mkdir(args.output_dir)
    except:
        pass
    
    convert_to_png_ = partial(convert_to_png,  dir_img=args.output_dir)
    
    parallel(convert_to_png_, list(Path(args.dicom_dir).iterdir()), max_workers=args.num_workers)
    
if __name__ == "__main__":
    main()
