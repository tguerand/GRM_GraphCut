# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:43:28 2021

@author: trist
"""

import os
import sys
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm

path_grid = r'grid_sizes_dataset.csv'
path_wkt = r'train_wkt_dataset.csv'

grid_df = pd.read_csv(path_grid)
wkt_df = pd.read_csv(path_wkt)

class Multigon():
    """A single polygon"""
    
    def __init__(self, imageId, classType, multipolygonWKT):
        """Args
        ------
        imageId: str
            the id of the image
        classType: int
            the class id
        multipolygonWKT : str
            the multiploygonWKT attribute
        
        
        Attributes
        ------
        imageId: str
            the id of the image
        classType: int
            the class id
        points : list
            the list of the polygons"""
        
        self.imageId = imageId
        self.classType = classType
        self.points = self.get_points(multipolygonWKT)
    
    def get_points(self, attribute):
        """Returns the list of the polygons from the string of the csv
        
        Args
        ------
        attribute: str
            the multiploygonWKT attribute
        
        Return
        -----
        points_list: list
            list of the polygons, in list format. Each points is in a list"
            polygon[-1] == polygon[0]"""
            
        points_list = attribute.replace('(', '[')
        points_list = points_list.replace(')', ']')
    
        if points_list.split(' ')[-1] == 'EMPTY':
            points_list = []
        else:
            points_list = '[' + points_list[13:] + ']'
            points_list = points_list.replace(', ','],[')
            points_list = points_list.replace(' ',',')
            points_list = ast.literal_eval(points_list)
            
            for i in range(len(points_list)):
                for j in range(len(points_list[i])):
                    
                    points_list[i][j] = np.array(points_list[i][j]) 
            
        return points_list
        
class Multigones():
    
    def __init__(self):
        
        self.multigones = []
    
    def get_multi(self, df):
        """Save all multigones of the train df into a list: the multigones attribute
        
        Args
        -----
        df: pandas.Dataframe
            the training dataframe
        """
        for i in tqdm(range(len(df.index))):
        
            self.multigones.append(Multigon(wkt_df['ImageId'][i],
                                            wkt_df['ClassType'][i],
                                            wkt_df['MultipolygonWKT'][i]))
            
        
        
m3 = Multigon(wkt_df['ImageId'][3], wkt_df['ClassType'][3], wkt_df['MultipolygonWKT'][3])
m250 = Multigon(wkt_df['ImageId'][25], wkt_df['ClassType'][25], wkt_df['MultipolygonWKT'][25])


#ms1 = Multigones()
#ms1.get_multi(wkt_df)



