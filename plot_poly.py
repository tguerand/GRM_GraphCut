# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:46:49 2021

@author: trist
"""

import os

import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np
import pandas as pd
import ast

from shapely.geometry.polygon import Polygon
from PIL import Image
from PIL import ImageDraw

from loader import Loader

def plot_poly(df, poly_idx, data_path='dataset', out_path='out.png'):
    
    po = ast.literal_eval(df['geom'].iloc[poly_idx])
    
    
    
    img_id = df['ImageId'].iloc[poly_idx]
    
    
    img = Image.open(os.path.join(data_path, img_id + '.tif'))
    
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    draw.polygon(po, fill = "wheat")
    
    img3 = Image.blend(img, img2, 0.5)
    img3.save('out.png')

if __name__ == '__main__':


    data_path = 'dataset' 
    df_path = 'df_with_polygons_as_pixels.csv'
    
    if not(os.path.exists(df_path)):
        Loader('train_wkt_dataset.csv',
               'grid_sizes_dataset.csv').save_final_df(out_path=df_path)
    
    df_poly = pd.read_csv(df_path)
    
    plot_poly(df_poly, 60000)
        
    
    
