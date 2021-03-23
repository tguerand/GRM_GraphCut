import os
import pandas as pd
import re
import shutil

grid_sizes = pd.read_csv('grid_sizes.csv', header=0, sep=',')
train_wkt = pd.read_csv('train_wkt_v4.csv', header=0, sep=',')

image_ids = []
for image_id in train_wkt.drop_duplicates('ImageId').ImageId:
    image_ids.append(image_id)
    shutil.copy(f'three_band/{image_id}.tif', f'dataset/{image_id}.tif')


grid_sizes_dataset = grid_sizes[grid_sizes.ImageId.isin(image_ids)]
grid_sizes_dataset.to_csv('grid_sizes_dataset.csv', index=False)

train_wkt_dataset = train_wkt[train_wkt.ImageId.isin(image_ids)]
train_wkt_dataset.to_csv('train_wkt_dataset.csv', index=False)
