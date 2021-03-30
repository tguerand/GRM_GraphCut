from PIL import Image
import numpy as np

from apply_min_cut import apply_min_cut
from graph import create_graph_from_images
import matplotlib.pyplot as plt

from viz import crop
import os
import cv2

def binary_image_segmentation(image_path, gamma, fore, back):
    I = Image.open(image_path)  # read image
    I = np.array(I)

    plt.imshow(I)
    plt.show()

    graph = create_graph_from_images(I, gamma, fore, back, _polys='binary')

    m, n, _ = I.shape
    source_node = m*n
    sink_node = m*n+1

    partition = apply_min_cut(graph.graph, source_node, sink_node)
    reachable, non_reachable = partition

    output_img = np.zeros(I.shape)

    for i in range(m):
        for j in range(n):
            if i*n+j in non_reachable:
                output_img[i, j, 0], output_img[i, j, 1], output_img[i, j, 2] = I[i, j, 0], I[i, j, 1], I[i, j, 2]
            else:
                output_img[i, j, 0], output_img[i, j, 1], output_img[i, j, 2] = 255, 255, 255

    return output_img


def get_rectangle(upper, left, down, right):
    rectangle = []
    for i in range(upper, down):
        for j in range(left, right):
            rectangle.append([i, j])

    return rectangle


if __name__ == '__main__':
    gamma = 0.01
    # fore = get_rectangle(50, 250, 100, 350)
    # back = get_rectangle(130, 0, 230, 100)
    
    img_path = os.path.join(r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\dataset\jpg_img',
                            '6110_4_0.jpg')
    
    img_id = os.path.split(img_path)[-1].split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.imread(img_path)
    img_dir = r'./dataset/jpg_img'
    
    alpha = 3
    
    lbl_path_a = os.path.join(img_dir, 'polys', img_id + '_' + str(alpha) + '.jpg')
    lbl_img_a = cv2.imread(lbl_path_a, cv2.IMREAD_GRAYSCALE)
    mask_a = np.abs(img - lbl_img_a)

    mask_a[np.where(mask_a>0)]=1
    all_poly = crop.mask_to_polygons(mask_a)
    fore = crop.shapely_to_xy(all_poly)
    
    back = np.where(mask_a==0)
    back = [[back[0][i], back[1][0]] for i in range(len(back[0]))]
    out_ = np.zeros(img.shape)
    out_[back] = 1
    #plt.imshow(out_)
    output_img = binary_image_segmentation(img_path, gamma, fore, back)

    plt.imshow(output_img.astype('uint8'))  # plot the output image
    plt.show()