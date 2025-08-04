import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
import torch

def Eigen(adj, d, adj_normalize, feature_abs):
    """
    Calculate the top-d eigenvectors of the adjacency matrix
    
    Input:
    adj: adjacency matrix in scipy sparse format
    d: the dimensionality of the eigenspace
    adj_normalize: whether to symmetrically normalize adj
    feature_abs: whether to use absolute function 
    
    Output:
    X: n * d numpy array
    """
    if adj_normalize:
        adj = normalize_adj(adj)
    
    # Convert adjacency matrix to PyTorch sparse tensor
    row, col = adj.nonzero()
    indices = torch.LongTensor(np.stack((row, col), axis=0)).to('cuda:0')
    values = torch.FloatTensor(adj.data).to('cuda:0')
    shape = torch.Size(adj.shape)
    adj_torch = torch.sparse_coo_tensor(indices, values, shape).to('cuda:0')
    
    # Perform SVD using PyTorch
    u, s, vt = torch.svd(adj_torch.to_dense())
    s = s.cpu().numpy()
    u = u.cpu().numpy()
    vt = vt.cpu().numpy()  # Get vt as well, though it's not used here
    
    # Compute eigenvalues from singular values
    lamb = s**2
    
    # Sort eigenvalues and select the top d
    indices_sorted = np.argsort(lamb)[::-1]  # Sort in descending order
    top_indices = indices_sorted[:d]
    
    # Select top d eigenvectors
    X = u[:, top_indices]
    
    # Sort eigenvectors according to eigenvalues
    lamb = lamb[top_indices]
    X = X[:, np.argsort(lamb)]
    
    if feature_abs:
        X = np.abs(X)
    else:
        # Ensure sign consistency of eigenvectors
        for i in range(X.shape[1]):
            if X[np.argmax(np.absolute(X[:, i])), i] < 0:
                X[:, i] = -X[:, i]
    
    return X

def Eigen_multi(adj, d, adj_normalize, feature_abs):
    """
    Handle if the graph has multiple connected components
    Arguments are the same as Eigen    
    """
    G = nx.from_scipy_sparse_matrix(adj)
    comp = list(nx.connected_components(G))
    X = np.zeros((adj.shape[0],d))
    for i in range(len(comp)):
        node_index = np.array(list(comp[i]))
        d_temp = min(len(node_index) - 2, d)
        if d_temp < 1:
            continue
        adj_temp = adj[node_index,:][:,node_index].asfptype()
        X[node_index,:d_temp] = Eigen(adj_temp, d_temp, adj_normalize, feature_abs)
    return X

def normalize_adj(adj):
    """ Symmetrically normalize adjacency matrix."""
    """ Copy from https://github.com/tkipf/gcn """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    
    # Avoid divide by zero by replacing zero values in rowsum with a very small number
    rowsum = np.maximum(rowsum, 1e-8)  # Replace 0 with a small positive number
    
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()