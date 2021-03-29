import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    def __init__(self, sink_node):
        self.source_node = sink_node - 1
        self.sink_node = sink_node
        self.graph = nx.DiGraph()
        self.ROW = sink_node + 1

    def add_terminal_edge(self, node, source_weight, sink_weight):
        self.graph.add_edge(self.source_node, node, capacity=source_weight)
        self.graph.add_edge(node, self.sink_node, capacity=sink_weight)

    def add_edge(self, node1, node2, weight):
        self.graph.add_edge(node1, node2, capacity=weight)


def get_log_or_100(value):
    return -np.log(value) if value != 0 else 100000


def get_value_for_a_channel(pixels_value, channel):
    return [pixel_value[channel] for pixel_value in pixels_value]


def create_histogram_for_label(pixels_value):
    hist_r, _ = np.histogram(get_value_for_a_channel(pixels_value, 0), density=True, bins=257,
                             range=(0, 256))
    hist_g, _ = np.histogram(get_value_for_a_channel(pixels_value, 1), density=True, bins=257,
                             range=(0, 256))
    hist_b, _ = np.histogram(get_value_for_a_channel(pixels_value, 2), density=True, bins=257,
                             range=(0, 256))

    #plt.plot(hist_r, color='red')
    #plt.plot(hist_g, color='green')
    #plt.plot(hist_b, color='blue')
    #plt.show()

    return hist_r, hist_g, hist_b


def get_value_for_label(pixel, hist_r, hist_g, hist_b):
    pdf_with_f = np.array([get_log_or_100(hist_r[pixel[0]]),
                           get_log_or_100(hist_g[pixel[1]]),
                           get_log_or_100(hist_b[pixel[2]])])
    return np.sum(pdf_with_f)


def get_pixels_value_from_coord_list(I, coord_list):
    pixel_list = []
    for coord in coord_list:
        pixel_list.append(I[coord[0], coord[1]])
    return np.array(pixel_list)


def create_graph_from_images(I, gamma, fore_coord_list, back_coord_list):
    """
    Return graphs computed with source is foreground and sink is background
    :param I: np array image
    :param gamma: factor to compute on the exponential for weights between nodes
    :param fore_coord_list: list of pixels in the foreground
    :param back_coord_list: list of pixels in the background
    """
    If = get_pixels_value_from_coord_list(I, fore_coord_list)  # take a part of the foreground
    Ib = get_pixels_value_from_coord_list(I, back_coord_list)  # take a part of the background

    # Create histograms for each pixel
    hist_if_r, hist_if_g, hist_if_b = create_histogram_for_label(If)
    hist_ib_r, hist_ib_g, hist_ib_b = create_histogram_for_label(Ib)

    m, n = I.shape[0], I.shape[1]  # copy the size

    graph = Graph(m*n+1)

    # Define the probability with background and foreground
    F = np.zeros((m, n))
    B = np.zeros((m, n))

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):

            # Taking pixel values in hist
            pixel = I[i, j]
            F[i, j] = get_value_for_label(pixel, hist_if_r, hist_if_g, hist_if_b)
            # F[i, j] = get_log_or_100(hist_f, pixel)
            # B[i, j] = get_log_or_100(hist_, pixel)
            B[i, j] = get_value_for_label(pixel, hist_ib_r, hist_ib_g, hist_ib_b)

    for i in range(m):  # checking the 4-neighborhood pixels
        for j in range(n):
            node = i*n+j
            ws = F[i, j]
            wt = B[i, j]
            graph.add_terminal_edge(node, ws, wt)

            pixels_to_update = [(i, j-1), (i, j+1), (i-1, j), (i+1, j)]

            for pixel_to_update in pixels_to_update:
                node_pixel = pixel_to_update[0]*n + pixel_to_update[1]
                if 0 <= pixel_to_update[0] < m and 0 <= pixel_to_update[1] < n:
                    weight = 100*np.exp(-gamma * (np.linalg.norm(I[i, j] - I[pixel_to_update]) ** 2))
                    graph.add_edge(node, node_pixel, weight)

    return graph
