import tifffile as tiff
import matplotlib.pyplot as plt

path = r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\dataset\6010_1_2.tif'
P = tiff.imread(path)
tiff.imshow(P)
plt.axis('off')
plt.show()
