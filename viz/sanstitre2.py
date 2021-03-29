# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 19:19:53 2021

@author: trist
"""

import tifffile as tiff

import numpy as np
import matplotlib.pyplot as plt
import cv2


data_path = r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\dataset\6010_1_2.tif'
out = r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\viz\test.jpg'

P = tiff.imread(data_path)
r, g, b = P
P = np.array([b, r, g])
print(P.shape)
P = np.moveaxis(P, 0, -1)
print(P.shape)


with tiff.TiffFile(data_path) as tif:
    fh = tif.filehandle
    print(len(tif.pages))
    for page in tif.pages:
        # for index, (offset, bytecount) in enumerate(
        #     zip(page.dataoffsets, page.databytecounts)
        # ):
        #     fh.seek(offset)
        #     data = fh.read(bytecount)
        #     tile, indices, shape = page.decode(
        #         data, index, jpegtables=page.jpegtables
        #     )
        im = page.asarray()
        
plt.imsave(out, P.astype(np.uint8), cmap='brg', format='jpg')
#cv2.imwrite(out, P)