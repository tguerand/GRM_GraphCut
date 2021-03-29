# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 14:58:58 2021

@author: trist
"""

import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
from PIL import Image
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
    #plt.figure(figsize=(h/1000, w/1000), dpi=100)
    #image = tiff.imread(tiff_img) 
    tiff.imshow(image)
    
    ax = plt.gca()    
    ax.set_axis_off()
    
    plt.savefig(out_path, bbox_inches='tight',pad_inches = 0)#, dpi=1000)
    
    #plt.show()
    # image = np.moveaxis(image, 0, -1)
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(w,h)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(image)
    # fig.savefig('figure.png', dpi=1)


def fit_poly(img_path, df_path):
    """Match the polygone size to the jpg image
    
    Args
    -----
    img_path: str
        the path of the image
    df: pd.Dataframe
        the data df, where every polygone is saved
    """
    df = pd.read_csv(df_path)
    
    jpg_img = np.asarray(Image.open(img_path))
    img_id = img_path.split('.')[-2].split(r'/')[-1]
    
    for poly_idx in df['geom'][df['ImageId'] == img_id].index.tolist():
        name = img_id + 'red_poly.jpg'
        plot_poly(df, poly_idx, data_path='../dataset', out_path=name)
        jpg_poly = np.asarray(Image.open(name))
        mask = abs(jpg_img - jpg_poly)
        mask[np.where(mask > 0)] = 1
        img, contours, _ = cv2.findContours(mask, cv2.RER_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        coords = np.column_stack(np.where(mask > 0))
        print(coords)
        df['geom_red'][poly_idx] = coords
    print(df.columns)
    df.to_csv(df_path)

    
path = r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\dataset\6010_1_2.tif'
ratio = 0.75
jpg_path = '../dataset/jpg_img/6010_1_2.jpg'

#crop(path, ratio, jpg_path)

df_path = r'../df/df_with_polygons_as_pixels.csv'
    
if not(os.path.exists(df_path)):
    Loader('train_wkt_dataset.csv',
            'grid_sizes_dataset.csv').save_final_df(out_path=df_path)

#fit_poly(jpg_path, df_path)
df_poly = pd.read_csv(df_path)

dir_path =  r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\dataset'

for f in os.listdir(dir_path):
    if f[-3:]=='tif':
        name = r'jpg_img' + f[:-3]+'jpg'
        crop(os.path.join(dir_path, f), ratio, name)

