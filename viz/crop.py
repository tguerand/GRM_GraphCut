# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 14:58:58 2021

@author: trist
"""

import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas as pd
from plot_poly import plot_poly
from loader import Loader

def crop(tiff_img, ratio, out_path):
    """
    Crop a tiff image with a certain ration to reduce its size
    
    https://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels
    
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
    h, w, c = image.shape
    plt.figure(figsize=(h/1000, w/1000), dpi=100)
    #image = tiff.imread(tiff_img) 
    tiff.imshow(image)
    
    ax = plt.gca()    
    ax.set_axis_off()
    
    plt.savefig(out_path, bbox_inches='tight',pad_inches = 0, dpi=1000)
    
    #plt.show()
    # image = np.moveaxis(image, 0, -1)
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(w,h)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(image)
    # fig.savefig('figure.png', dpi=1)
    
path = r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\dataset\6010_1_2.tif'
ratio = 0.75
out_path = 'out.jpg'

crop(path, ratio, out_path)

# df_path = r'../df/df_with_polygons_as_pixels.csv'
    
# if not(os.path.exists(df_path)):
#     Loader('train_wkt_dataset.csv',
#            'grid_sizes_dataset.csv').save_final_df(out_path=df_path)

# df_poly = pd.read_csv(df_path)

# plot_poly(df, poly_idx, data_path='dataset', out_path='out.png')