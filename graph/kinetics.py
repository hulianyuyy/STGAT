import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

num_node = 18
inward_ori_index = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, K=3, sample_size=5):
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(neighbor, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(neighbor + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary_with_I)
        '''
        self.Ak = tools.k_total_adjacency(self.A_binary_with_I, K)
        
        self.Akt = tools.build_spatial_temporal_graph(self.Ak, sample_size)
        self.Akt = tools.normalize_adjacency_matrix(self.Akt)
        
        A = tools.build_spatial_temporal_graph(self.A_binary_with_I, sample_size)
        self.A_scales = [tools.k_adjacency(A, k, with_self=True) for k in range(K)]
        self.A_scales = np.concatenate([tools.normalize_adjacency_matrix(g) for g in self.A_scales])
        
        self.A_expand = tools.get_expanded_graph(num_node, inward, type=2)
        self.A_expand = tools.normalize_adjacency_matrix(self.A_expand)
        
        self.A_strenghtened = tools.get_strenghtened_graph(num_node, inward)
        self.A_strenghtened = tools.normalize_adjacency_matrix(self.A_strenghtened)
        '''
        #self.A_sep = tools.seperated_adjacency(self.A_binary_with_I, [0, 2, 4, 6, 8, 10, 12])
        self.A_sep = tools.seperated_adjacency(self.A_binary_with_I, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # max reachable length = 7



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary_with_I, A_sep0, A_sep1, A_sep2 = graph.A, graph.A_binary_with_I, graph.A_sep[0], graph.A_sep[1], graph.A_sep[2]
    
    f, ax = plt.subplots(1, 5)
    ax[0].imshow(A, cmap='gray')
    ax[1].imshow(A_binary_with_I, cmap='gray')
    ax[2].imshow(A_sep0, cmap='gray')
    ax[3].imshow(A_sep1, cmap='gray')
    ax[4].imshow(A_sep2, cmap='gray')
    plt.show()
    
    print('A_binary_with_I:',A_binary_with_I)
    print('A_sep1:',A_sep1)
