from _collections import deque
import numpy as np
from numba import jit

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
