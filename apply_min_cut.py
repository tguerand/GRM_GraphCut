from _collections import deque
import numpy as np
from numba import jit
import itertools

import graph as gr
#from graph import create_graph_from_image
import pandas as pd
import os
import cv2
import networkx as nx
from viz.crop import mask_to_polygons, shapely_to_xy


gamma = 0.001
fore = (225, 142, 279, 185)
back = (7, 120, 61, 163)

@jit(nopython=True)
def BFS(graph, s, t, parent):
    # Mark all the vertices as not visited
    ROW = len(graph[0])
    visited = [False] * ROW

    # Create a queue for BFS
    queue = deque(s)

    # Mark the source node as visited and enqueue it
    visited[s] = True

    # Standard BFS Loop
    while queue:

        # Dequeue a vertex from queue and print it
        u = queue.popleft()

        # Get all adjacent vertices of the dequeued vertex u
        # If a adjacent has not been visited, then mark it
        # visited and enqueue it
        for ind, val in enumerate(graph[u]):
            if visited[ind] is False and val > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u

    # If we reached sink in BFS starting from source, then return true, else false
    return True if visited[t] else False

@jit(nopython=True)
def FordFulkerson(graph, source, sink):
    """
    :param graph: graph with nodes and edges from create_graph_from_images
    :param source: source node
    :param sink: sink node
    :return: maximum flow from s to t in the given graph and residual graph
    """
    graph_copy = graph.copy()

    len_rows = len(graph_copy[0])

    # This array is filled by BFS and to store path
    parent = np.array([-1 for _ in range(len_rows)])

    max_flow = 0  # There is no flow initially

    # Augment the flow while there is path from source to sink
    while BFS(graph_copy, source, sink, parent):

        # Find minimum residual capacity of the edges along the
        # path filled by BFS. Or we can say find the maximum flow
        # through the path found.
        path_flow = float("Inf")
        s = sink
        while s != source:
            path_flow = min(path_flow, graph_copy[parent[s]][s])
            s = parent[s]

        # Add path flow to overall flow
        max_flow += path_flow

        # update residual capacities of the edges and reverse edges
        # along the path
        v = sink
        while v != source:
            u = parent[v]
            graph_copy[u][v] -= path_flow
            v = parent[v]

    return max_flow, graph_copy

def alpha_beta_swap(img_path, nb_pixel_min_change, alphas, gamma=gamma):
    """Performs the alphabeta swap graph cut algorithm
    
    Args
    -----
    img_path: str
        the path of the image
    nb_pixel_min_change: int
        threshold
    gamma: float
        factor to compute on the exponential for weights between nodes
        
    """
    img_id = os.path.split(img_path)[-1].split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.imread(img_path)
    img_dir = r'./dataset/jpg_img'
    # create the "other" label
    other = np.zeros(img.shape)
    
    poly_files = [file for file in os.listdir(os.path.join(img_dir, 'polys')) if img_id in file]
                                              
    for f in poly_files:
        lbl = cv2.imread(os.path.join(img_dir, 'polys', f), cv2.IMREAD_GRAYSCALE)
        other += lbl
    
    idx = np.where(other > 0)
    id0 = np.where(other == 0)
    
    other[idx] = img[idx].copy()
    other[id0] = img[id0].copy() + 50
    other[np.where(other > 255)] = 255

    other_path =  os.path.join(img_dir, 'polys', img_id +'_100.jpg')  
    cv2.imwrite(other_path, other)
    
    # create the tupples for alpha beta swap
    # first add the "other" label
    alphas.append(100)
    
    alphabeta = []
    for i in range(len(alphas)-1):
        for j in range(i+1, len(alphas)):
            alphabeta.append([alphas[i], alphas[j]])
    
    count = 0
    conv = nb_pixel_min_change + 10
    
    
    
    out_ = np.ones(img.shape) * alphabeta[-1][0]
    
    
    
    ##### TODO : il faut s occuper du label autre
    # soit le créer dans la fonction, soit le faire en amont dans le code
    
    print('nb of pixel minimum to change: {}'.format(nb_pixel_min_change))
    
    while conv > nb_pixel_min_change:
        print(count)
        alpha = alphabeta[i%len(alphabeta)][0]
        beta = alphabeta[i%len(alphabeta)][1]
        # define the list of pixels, now it is a rectangle --> mask
        lbl_path_a = os.path.join(img_dir, 'polys', img_id + '_' + str(alpha) + '.jpg')
        lbl_img_a = cv2.imread(lbl_path_a, cv2.IMREAD_GRAYSCALE)
        mask_a = np.abs(img - lbl_img_a)
    
        mask_a[np.where(mask_a>0)]=1
        all_poly = mask_to_polygons(mask_a)
        poly_a = shapely_to_xy(all_poly)
        
        lbl_path_b = os.path.join(img_dir, 'polys', img_id + '_' + str(beta) + '.jpg')
        lbl_img_b = cv2.imread(lbl_path_b, cv2.IMREAD_GRAYSCALE)
        mask_b = np.abs(img - lbl_img_b)
        
        mask_b[np.where(mask_b>0)]=1
        all_poly = mask_to_polygons(mask_b)
        poly_b = shapely_to_xy(all_poly)
        
        
        g = gr.create_graph_from_images(img_rgb, gamma, poly_a, poly_b, _polys='multiple')
        
       
        
        m, n = img.shape
        # number_of_nodes method returns len of list
        source_node = g.graph.number_of_nodes() - 2
        sink_node = g.graph.number_of_nodes() - 1
    
        _, partition = nx.minimum_cut(g.graph, source_node, sink_node)
        reachable, non_reachable = partition
    
        output_img_a = np.zeros(img.shape)
        output_img_b = np.zeros(img.shape)
    
        for i in range(m):
            for j in range(n):
                if i*n+j in non_reachable:
                    output_img_a[i, j] = 0
                    output_img_b[i, j] = 1
                else:
                    output_img_a[i, j] = 1
                    output_img_b[i, j] = 0
                    
        
        nb_a = np.sum(np.abs(mask_a - output_img_a))
        
        out_[np.where(output_img_a == 1)] = alpha
        out_[np.where(output_img_b == 1)] = beta
        conv  = nb_a
        print('We changed : {} pixels'.format(conv))
        
        count += 1
    return out_
        
    
if __name__ == '__main__':
    
    img_path = os.path.join(r'C:\Users\trist\Documents\CS\3A\GRM\GRM_GraphCut\GRM_GraphCut\dataset\jpg_img',
                            '6110_4_0.jpg')
    nb_pixel_min_change = 1000
    alphas = [3, 8]
    
    out = alpha_beta_swap(img_path, nb_pixel_min_change, alphas)