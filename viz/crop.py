# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 14:58:58 2021

@author: trist
"""

import tifffile as tiff
import matplotlib.pyplot as plt
import cv2

def crop(tiff_img, ratio, out_path):
    """
    Crop a tiff image with a certain ration to reduce its size
    
    Args
    -----
    tiff_img: str
        the path of the image
    ratio: float
        the ratio to save the image, eg 0.75 will reduce its size to 0.75 of
        the original size
    out_path: str
        the pave to save the cropped img
    """
    
    
    
    image = tiff.imread(tiff_img) 
    tiff.imshow(image)
    
    ax = plt.gca()    
    ax.set_axis_off()
    
    plt.savefig(out_path, bbox_inches='tight',pad_inches = 0)
    
    plt.show()
    
path = r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\dataset\6010_1_2.tif'
ratio = 0.75
out_path = 'out.jpg'

crop(path, ratio, out_path)