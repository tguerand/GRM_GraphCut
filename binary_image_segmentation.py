from PIL import Image
import numpy as np

from apply_min_cut import apply_min_cut
from graph import create_graph_from_images
import matplotlib.pyplot as plt


def binary_image_segmentation(image_path, gamma, fore, back):
    I = Image.open(image_path)  # read image
    I = np.array(I)

    plt.imshow(I)
    plt.show()

    graph = create_graph_from_images(I, gamma, fore, back)

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
    gamma = 0.001
    fore = get_rectangle(50, 250, 100, 350)
    back = get_rectangle(130, 0, 230, 100)
    output_img = binary_image_segmentation('img_cow_best.jpeg', gamma, fore, back)

    plt.imshow(output_img.astype('uint8'))  # plot the output image
    plt.show()
