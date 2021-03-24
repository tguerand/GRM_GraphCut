# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:46:49 2021

@author: trist
"""

import os

import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np

from loader import Loader

if __name__ == '__main__':


    data_path = 'dataset' 
    Loader('train_wkt_dataset.csv', 'grid_sizes_dataset.csv').save_final_df()
