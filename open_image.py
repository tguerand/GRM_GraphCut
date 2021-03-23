import tifffile as tiff
import matplotlib.pyplot as plt


P = tiff.imread('dataset/6070_2_3.tif')
tiff.imshow(P)
plt.show()
