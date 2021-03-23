##TO DO :
- I think that a class is not needed, functions are enough (maybe a class later for the whole model)
- use numba
    - use numpy instead of lists when possible
    - add @jit(nopython=True) before functions
    - output 
- build graph cut from FordFulkerson
    - how to cut on the max flow ?
    - assign a label to each subgraph
    - compute cost from the original graph
- build multi-label graph cut
    - alpha expression method
    - alpha beta swap
- build the graphs of images
    - structure : source + 2D grid with 4 neighbours of pixels + sink
    - weights :
        - local preferrence : from statistics computed on the prelabelled images
        - interaction cost : cost to move from a class to another one (can be hyper parameter or determined by the statistics done previousely)


## Ford Fulkerson 
From https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
