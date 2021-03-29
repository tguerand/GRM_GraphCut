import cv2
import numpy as np
from PIL import Image

from apply_min_cut import FordFulkerson


class Graph:
    def __init__(self, sink_node):
        self.source_node = sink_node - 1
        self.sink_node = sink_node
        self.graph = np.zeros((sink_node + 1, sink_node + 1))
        self.ROW = sink_node + 1

    def add_terminal_edge(self, node, source_weight, sink_weight):
        self.graph[self.source_node][node] = source_weight
        self.graph[node][self.sink_node] = sink_weight

    def add_edge(self, node1, node2, weight):
        self.graph[node1][node2] = weight


def create_graph_from_images(img_path, gamma, fore, back, _datatype='pixels'):
    """
    Return graphs computed with source is foreground and sink is background
    :param img_path: Img path
    :param gamma: factor to compute on the exponential for weights between nodes
    :param fore: area of the foreground in the image to compute weight between node and source
    :param back: area of the background in the image to compute weight between node and sink
    :return:
    """
    I = Image.open(img_path).convert('L')  # read image

    If = np.array(I.crop(fore))  # take a part of the foreground
    Ib = np.array(I.crop(back))  # take a part of the background
    I = np.array(I)

    hist_if = cv2.calcHist([If], [0], None, [256], [0, 256])
    hist_ib = cv2.calcHist([Ib], [0], None, [256], [0, 256])

    Ifmean = np.argmax(hist_if)  # get argmax of the histogram for foreground
    Ibmean = np.argmax(hist_ib)  # get argmax of the histogram for background

    Im = I.copy().flatten().astype('int')  # Converting the image array to a vector for ease.
    m, n = I.shape[0], I.shape[1]  # copy the size

    graph = Graph(m*n+1)

    # Define the probability with background and foreground
    F = np.zeros(I.shape)
    B = np.zeros(I.shape)

    for i in range(I.shape[0]):  # Defining the Probability function....
        for j in range(I.shape[1]):
            diff_back = abs(I[i, j] - Ifmean)
            diff_fore = abs(I[i, j] - Ibmean)
            den = diff_back + diff_fore

            # Probability of a pixel being foreground
            if diff_fore > 0:
                F[i, j] = -np.log(diff_fore / den)
            else:
                F[i, j] = 100

            # Probability of a pixel being background
            if diff_back > 0:
                B[i, j] = -np.log(diff_back / den)
            else:
                B[i, j] = 100

    F, B = F.flatten(), B.flatten()  # convertingb  to column vector for ease

    for i in range(m * n):  # checking the 4-neighborhood pixels
        ws = (F[i] / (F[i] + B[i]))  # source weight
        wt = (B[i] / (F[i] + B[i]))  # sink weight
        graph.add_terminal_edge(i, ws, wt)

        pixels_to_update = [i - 1, i + 1, i - n, i + n]

        for pixel_to_update in pixels_to_update:
            if 0 <= pixel_to_update < m * n:
                weight = np.exp(-gamma * (abs(Im[i] - Im[pixel_to_update]) ** 2))
                graph.add_edge(i, pixel_to_update, weight)

    return graph


if __name__ == '__main__':
    gamma = 0.001
    fore = (225, 142, 279, 185)
    back = (7, 120, 61, 163)
    g = create_graph_from_images('input1.jpeg', gamma, fore, back)
    print(FordFulkerson(g.graph, len(g.graph[0])-2, len(g.graph[0])-1))
