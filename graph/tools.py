import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_expanded_graph(num_nodes, inward, type=1):
    A_expand = np.zeros((num_nodes + 6, num_nodes + 6), dtype=np.float32)       # 25: upper_left,  26: upper_right,  27: up,  28: middle, 29: lower_left, 30: lower_right
    if type ==1:
        '''
        inward = inward + [(25, 4), (25, 5), (25, 6), (25, 7), (25, 21), (25, 22), (25, 26), (25, 27)]
        inward = inward + [(26, 8), (26, 9), (26, 10), (26, 11), (26, 23), (26, 24), (26, 27)]
        inward = inward + [(27, 2), (27, 3)]
        inward = inward + [(28, 0), (28, 1), (28, 20), (28, 25), (28, 26), (28, 27), (28, 29), (28, 30)]
        inward = inward + [(29, 12),(29, 13),(29, 14),(29, 15),(29, 30)]
        inward = inward + [(30, 16),(30, 17),(30, 18),(30, 19)]
        '''
        inward = inward + [(25, 26), (25, 27)]
        inward = inward + [(26, 27)]
        inward = inward + [(28, 25), (28, 26), (28, 27), (28, 29), (28, 30)]
        inward = inward + [(29, 30)]
        outward = [(j, i) for (i, j) in inward]
        edges = inward + outward
    elif type ==2:
        inward = inward + [(25, 4), (25, 5), (25, 6), (25, 7), (25, 21), (25, 22), (25, 26), (25, 27)]
        inward = inward + [(26, 8), (26, 9), (26, 10), (26, 11), (26, 23), (26, 24), (26, 27)]
        inward = inward + [(27, 2), (27, 3)]
        inward = inward + [(28, 0), (28, 1), (28, 20), (28, 25), (28, 26), (28, 27), (28, 29), (28, 30)]
        inward = inward + [(29, 12),(29, 13),(29, 14),(29, 15),(29, 30)]
        inward = inward + [(30, 16),(30, 17),(30, 18),(30, 19)]
        inward.remove((16, 0))
        inward.remove((12, 0))
        inward.remove((4, 20))
        inward.remove((8, 20))
        #inward.remove((2, 20))
        outward = [(j, i) for (i, j) in inward]
        edges = inward + outward
    for edge in edges:
        A_expand[edge] = 1.
    A_expand = A_expand + np.eye(len(A_expand), dtype=np.float32)
    return A_expand
    
def get_strenghtened_graph(num_nodes, inward):
    A_strenghtened = np.zeros((num_nodes, num_nodes), dtype=np.float32)       
    inward = inward + [(14, 12), (15, 12)]
    inward = inward + [(19, 16), (18, 16)]
    inward = inward + [(21, 4), (22, 4), (6, 4), (7, 4)]
    inward = inward + [(23, 8), (24, 8), (10, 8), (11, 8)]
    outward = [(j, i) for (i, j) in inward]
    edges = inward + outward
    for edge in edges:
        A_strenghtened[edge] = 1.
    A_strenghtened = A_strenghtened + np.eye(len(A_strenghtened), dtype=np.float32)
    return A_strenghtened
    
def build_spatial_temporal_graph(A_binary, sample_size):
    assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
    # Build spatial-temporal graph
    A_large = np.tile(A_binary, (sample_size, sample_size)).copy()
    return A_large

def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak
    
def seperated_adjacency(A, k_list, with_self=True):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    Ak = []
    for i in range(len(k_list)-1):
        k1 = k_list[i]
        k2 = k_list[i+1]
        Ak.append(np.minimum(np.linalg.matrix_power(A , k2), 1) - np.minimum(np.linalg.matrix_power(A , k1), 1) + I)
    return np.concatenate([np.expand_dims(A,0) for A in Ak], 0)
    
def k_total_adjacency(A, k):
    assert isinstance(A, np.ndarray)
    if k == 0:
        I = np.eye(len(A), dtype=A.dtype)
        return I
    Ak = np.minimum(np.linalg.matrix_power(A, k), 1) 
    return Ak


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def get_adjacency_matrix(edges, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        A[edge] = 1.
    return A