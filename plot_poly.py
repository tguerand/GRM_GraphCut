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
    """Plot a single polygon of an image
    
    Args
    -----
    df: pd.Dataframe
        the data dataframe
        Cols : ['ImageId', 'geom', 'ClassType', 'Xmax', 'Ymin']
    poly_idx: int
        the idx of the row of the poly in df
    data_path: str
        path where the dataset is stored
    out_path: str
        path where the output image is saved"""
    po = ast.literal_eval(df['geom'].iloc[poly_idx])
    
    
    
    img_id = df['ImageId'].iloc[poly_idx]
    
    
    img = Image.open(os.path.join(data_path, img_id + '.tif'))
    
    
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    draw.polygon(po, fill = "wheat")
    
    img3 = Image.blend(img, img2, 0.5)
    img3.save(out_path)
    
def plot_polys(df, class_type, img_id, data_path='dataset', out_path='out.png'):
    """Plot all the polygons of a single image of a single class
    
    Args
    -----
    df: pd.Dataframe
        the data dataframe
        Cols : ['ImageId', 'geom', 'ClassType', 'Xmax', 'Ymin']
    class_type: int
        the number id of the class to show
    img_id: str
        the id of the image
    data_path: str
        path where the dataset is stored
    out_path: str
        path where the output image is saved"""
    polys = []
    
    for pol in df['geom'][(df['ImageId']==img_id) & (df['ClassType']==class_type)].valuesgi:
        polys.append(ast.literal_eval(pol))
    
    img = Image.open(os.path.join(data_path, img_id + '.tif'))
    
    
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    for po in polys:
        draw.polygon(po, fill = "wheat")
    
    img3 = Image.blend(img, img2, 0.5)
    img3.save(out_path)
    

if __name__ == '__main__':


    data_path = 'dataset' 
    df_path = 'df_with_polygons_as_pixels.csv'
    
    if not(os.path.exists(df_path)):
        Loader('train_wkt_dataset.csv',
               'grid_sizes_dataset.csv').save_final_df(out_path=df_path)
    
    df_poly = pd.read_csv(df_path)
    
    plot_polys(df_poly, 4, '6040_2_2')
        
    
    
