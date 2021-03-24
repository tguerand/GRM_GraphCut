import tifffile as tiff
import matplotlib.pyplot as plt


P = tiff.imread('dataset/6170_4_1.tif')
tiff.imshow(P)
plt.show()
