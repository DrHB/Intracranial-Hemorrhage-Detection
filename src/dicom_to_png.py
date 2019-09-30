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
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def convert_to_png(dcm_in, _, dir_img):
    dcm = pydicom.dcmread(str(dcm_in))
    window_center, window_width, intercept, slope = get_windowing(dcm)
    img = pydicom.read_file(str(dcm_in)).pixel_array
    img = window_image(img, window_center, window_width, intercept, slope)
    cv2.imwrite(os.path.join(dir_img, os.path.basename(dcm_in)[:-3] + 'png'), img)
    
    

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
