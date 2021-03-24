import tifffile as tiff
import matplotlib.pyplot as plt


P = tiff.imread('dataset/6040_2_2.tif')
tiff.imshow(P)
plt.show()
