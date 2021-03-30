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
from shapely.geometry import Polygon, MultiPolygon
from collections import defaultdict

import matplotlib.patches
from matplotlib.collections import PatchCollection


from tqdm import tqdm
import os
import pandas as pd
import ast

#import plot_poly
#from loader import Loader

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


def fit_poly(img_path, df_path, dir_path, threshold=25):
    """Match the polygone size to the jpg image
    
    Args
    -----
    img_path: str
        the path of the image
    df: pd.Dataframe
        the data df, where every polygone is saved
    """
    df = pd.read_csv(df_path)
    
    jpg_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_id = img_path.split('.')[-2].split(r'/')[-1]
    
    df = df.astype({'geom_red':object})
    
    
    for poly_idx in tqdm(df['geom'][df['ImageId'] == img_id].index.tolist()):
        name = os.path.join(dir_path, img_id + 'red_poly.jpg')
        plot_poly.plot_poly(df, poly_idx, data_path='../dataset/tiff', out_path=name)
        jpg_poly = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.absdiff(jpg_img, jpg_poly)
        
        polygones = mask_to_polygons(mask) 
        polys = list(polygones)
        for i in range(len(polys)):
            polys[i] = polys[i].exterior.coords.xy
            polys[i] = [ [polys[i][0][j],
                          polys[i][1][j] ] for j in range(len( polys[i][0]))]
        df['geom_red'][poly_idx] = polys
        # arr_str = df['geom_red'][poly_idx]
        # print(arr_str)
        # arr_str = arr_str.replace('\n', '')[1:-1]
        # arr_l = arr_str.split('array')[1:]
        # for i in range(len(arr_l)):
        #     arr_l[i] = ast.literal_eval(arr_l[i].replace(',dtype=int32', ''))
        #     arr_l[i] = arr_l[i][0][0]
        # print(arr_l)
        
    
    df[['ImageId', 'geom', 'ClassType', 'Xmax', 'Ymin', 'geom_red']].to_csv(df_path, index=False)

  
    
def mask_to_polygons(mask, epsilon=0.5, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]

    if not approx_contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # need to re add in the check for type of all_polygons
        all_polygons = MultiPolygon(all_polygons)
    return all_polygons    
    

def shapely_to_xy(all_polygons):
    
    polys = list(all_polygons)
    for i in range(len(polys)):
        polys[i] = polys[i].exterior.coords.xy
        polys[i] = [ [int(polys[i][0][j]),
                      int(polys[i][1][j]) ] for j in range(len( polys[i][0]))]
    return polys


if __name__ == '__main__':

    
    dir_path =  r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\dataset\jpg_img'
    path = r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\dataset\6100_2_2.tif'
    ratio = 0.75
    
    
    jpg_path = '../dataset/jpg_img/6110_4_0.jpg'
    img_id = jpg_path.split('/')[-1].split('.')[0]
    class_type = 3
    
    
    #crop(path, ratio, jpg_path)
    
    df_path = r'../df/df_with_polygons_as_pixels.csv'
        
    if not(os.path.exists(df_path)):
        Loader('train_wkt_dataset.csv',
                'grid_sizes_dataset.csv').save_final_df(out_path=df_path)
    
    
    #fit_poly(jpg_path, df_path, dir_path)
    
    df = pd.read_csv(df_path)
    
    name = os.path.join(dir_path,'polys', img_id + '_'+ str(class_type) + '.jpg')
    #plot_poly(df, 450, data_path='../dataset/tiff', out_path=name)
    jpg_poly = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    jpg_img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.absdiff(jpg_img, jpg_poly)
    mask[np.where(mask > 0)] = 1
    plt.imshow(mask)
    plt.show()
    
    polygones = list(mask_to_polygons(mask))
    polygones = mask_to_polygons(mask) 
    polys = list(polygones)
    for i in range(len(polys)):
        polys[i] = polys[i].exterior.coords.xy
        polys[i] = [ [int(polys[i][0][j]),
                      int(polys[i][1][j]) ] for j in range(len( polys[i][0]))]
    
    
    #polys = []
    patches = [matplotlib.patches.Polygon(i) for i in polys]
    
    
    
    # for pol in df['geom_red'][(df['ImageId']==img_id) & (df['ClassType']==class_type)].values:
    #     print(pol)
    #     if pol == '[]':
    #         continue
    #     for poo in ast.literal_eval(pol):
            
    #         polys.append(poo)
        
    #         patches.append(matplotlib.patches.Polygon(polys[-1]))
    
    
    out_path = 'test.jpg'
    
    plt.imshow(jpg_img)
    
    ax = plt.gca()
    
    colors = 100 * np.random.rand(len(patches))
    p = PatchCollection(patches, alpha=0.4)
    p.set_array(colors)
    ax.add_collection(p)
    
    ax.set_axis_off()
    
    plt.savefig(out_path, bbox_inches='tight',pad_inches = 0)
    
    plt.show()
    plt.clf()
    
    
