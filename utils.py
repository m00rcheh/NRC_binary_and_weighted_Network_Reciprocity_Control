import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import rand
import random
import math
import seaborn as sb
from scipy.io import savemat
from numba import jit
import math
from scipy.linalg import schur
from scipy.sparse.csgraph import laplacian
import os
import community as community_louvain
import bct
from scipy.linalg import norm
# import utils


def convert_graph_to_adjacency_matrix(g):
    return nx.adjacency_matrix(g).toarray()


def convert_adjacency_matrix_to_graph(w):
    return nx.from_numpy_array(w)



def adjust_reciprocity_binary(adj_matrix, desired_reciprocity):
    np.fill_diagonal(adj_matrix, 0)
    adj_matrix = adj_matrix.astype('bool')
    L = adj_matrix.sum()
    current_r = compute_reciprocity_binary(adj_matrix)
    if np.isclose(current_r, desired_reciprocity, atol=1e-5):
        pass
    else:
        symmetric_indices = np.column_stack(np.where((adj_matrix == 1) * (adj_matrix.T == 1) *
                                                     np.triu(np.ones_like(adj_matrix), k=1)))
        num_symmetric_indices = symmetric_indices.shape[0]  # the network has num_symmetric_indices*2 reciprocal link
        num_reciprocal = int(desired_reciprocity * L)  # desired number of reciprocal links

        diff = 2 * num_symmetric_indices - num_reciprocal
        symmetric_zeros = np.column_stack(np.where((adj_matrix == 0) * (adj_matrix.T == 0) *
                                                   np.triu(np.ones_like(adj_matrix), k=1)))

        if diff >= 0:
            if symmetric_zeros.shape[0] < np.abs(diff):
                print('Error: density is too high. Sparsify your network and try again.')
            else:
                # remove reciprocal links
                num_to_remove = num_symmetric_indices - int(num_reciprocal/2)# - 1
                selected_indices = np.random.choice(num_symmetric_indices, num_to_remove, replace=False)
                selected_ones_indices = symmetric_indices[selected_indices[:num_to_remove]]
                for index in selected_ones_indices:
                    i, j = index
                    if np.random.rand(1, 1) >= 0.5:
                        adj_matrix[i, j] = 0
                        adj_matrix[j, i] = 1
                    else:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 0
                symmetric_zeros = np.column_stack(np.where((adj_matrix == 0) * (adj_matrix.T == 0) *
                                                           np.triu(np.ones_like(adj_matrix), k=1)))
                if symmetric_zeros.shape[0] >= int(num_to_remove):
                    selected_zero_indices_extra = np.random.choice(symmetric_zeros.shape[0],
                                                                   int(num_to_remove),
                                                                   replace=False)
                    selected_zero_to_ones_indices = symmetric_zeros[selected_zero_indices_extra]
                    for index in selected_zero_to_ones_indices:
                        i, j = index
                        if np.random.rand(1, 1) >= 0.5:
                            adj_matrix[i, j] = 0
                            adj_matrix[j, i] = 1
                        else:
                            adj_matrix[i, j] = 1
                            adj_matrix[j, i] = 0
        else:
            num_single_to_reciprocal = np.abs(diff/2) + 1
            Rest = np.abs(adj_matrix.astype('bool') ^ (adj_matrix.astype('bool')).T)
            num_single = 0.5 * np.abs(Rest).sum()
            if num_single >= 2 * num_single_to_reciprocal:
                single_links_indices = np.column_stack(np.where((adj_matrix == 0) * (adj_matrix.T == 1)))
                selected_single_link_indices = np.random.choice(int(num_single), int(2 * num_single_to_reciprocal),
                                                                replace=False)
                selected_links_indices = (
                    single_links_indices)[selected_single_link_indices[:int(num_single_to_reciprocal)]]
                for index in selected_links_indices:
                    i, j = index
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
                selected_links_indices_to_remove = single_links_indices[
                    selected_single_link_indices[int(num_single_to_reciprocal):]]
                for index in selected_links_indices_to_remove:
                    i, j = index
                    adj_matrix[i, j] = 0
                    adj_matrix[j, i] = 0
            else:
                single_links_indices = np.column_stack(np.where((adj_matrix == 0) * (adj_matrix.T == 1)))
                for index in single_links_indices:
                    i, j = index
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
                num_reciprocal_to_remove = int(single_links_indices.shape[0] / 2)
                symmetric_indices = np.column_stack(np.where((adj_matrix == 1) * (adj_matrix.T == 1) *
                                                            np.triu(np.ones_like(adj_matrix), k=1)))
                selected_indices = np.random.choice(symmetric_indices.shape[0], num_reciprocal_to_remove, replace=False)
                selected_ones_indices = symmetric_indices[selected_indices[:num_reciprocal_to_remove]]
                for index in selected_ones_indices:
                    i, j = index
                    adj_matrix[i, j] = 0
                    adj_matrix[j, i] = 0
                # print('Error: not enough single links to reciprocate. Run advanced algorithm')
    return adj_matrix


@jit(nopython=True)
def calculate_W1_and_r(W, L):
    np.fill_diagonal(W, 0)
    W1 = np.minimum(W, W.T)
    r = W1.sum() / L
    return W1, r


def compute_reciprocity_weighted(adj_matrix):
    np.fill_diagonal(adj_matrix, 0)
    L = adj_matrix.sum()
    W1 = np.minimum(adj_matrix, adj_matrix.T)
    # W2 = adj_matrix - W1
    return np.round(W1.sum() / L, 2)


@jit(nopython=True)
def adjust_matrix_gradually(W, L, target_r_new, increment=0.01, max_iter=1000, max_value=10, min_value=0.01):
    n = W.shape[0]
    current_W = W.copy()

    for _ in range(max_iter):
        # Compute current W1 and r
        current_W1, current_r = calculate_W1_and_r(current_W, L)

        if np.isclose(current_r, target_r_new, atol=1e-5) and np.abs(current_W.sum() - L) / L < 1e-5:
            break

        # Compute scaling factor for the current step
        if current_r == 0:
            scaling_factor = 0
        else:
            scaling_factor = target_r_new / current_r

        # Gradual scaling
        if scaling_factor > 1:
            scaling_factor = min(scaling_factor, 1 + increment)
        elif scaling_factor < 1:
            scaling_factor = max(scaling_factor, 1 - increment)

        # Apply scaling factor to elements contributing to W1
        W1_contributing_indices = np.argwhere(current_W <= current_W.T)

        for i, j in W1_contributing_indices:
            if i != j:
                current_W[i, j] = min(max_value, current_W[i, j] * scaling_factor)

        # Update W1 after scaling
        current_W1, current_r = calculate_W1_and_r(current_W, L)

        # Adjust non-contributing elements to maintain the total sum L
        total_increment = current_W.sum() - L
        if np.abs(total_increment) > 1e-5:
            non_contributing_indices = np.argwhere(current_W > current_W.T)
            np.random.shuffle(non_contributing_indices)
            total_decrement = 0
            for i, j in non_contributing_indices:
                if total_decrement >= total_increment:
                    break
                if current_W[i, j] > min_value:
                    decrease = min(current_W[i, j] - min_value,
                                   (total_increment - total_decrement) / len(non_contributing_indices))
                    current_W[i, j] -= decrease
                    total_decrement += decrease

    # Ensure final adjustment
    # if np.abs(current_W.sum() - L) > 1e-5:
    #     raise ValueError("Sum of W has changed!")
    return current_W


def adjust_matrix(W, L, target_r_new):
    current_W1, current_r = calculate_W1_and_r(W, L)
    scaling_factor = target_r_new / current_r
    v = np.float64(scaling_factor * current_W1)
    residual = current_W1.sum() - v.sum()
    current_W2 = W - current_W1
    positive_indices = current_W2 > 0
    count = np.sum(positive_indices)
    current_W2[positive_indices] += residual / count
    current_W = v + current_W2

    return current_W


def adjust_reciprocity_weighted(adj_matrix, desired_reciprocity, num_iter=1000):
    np.fill_diagonal(adj_matrix, 0)
    L = adj_matrix.sum()
    W1, r = calculate_W1_and_r(adj_matrix, L)
    if desired_reciprocity <= r:
        if np.isclose(r, 1.0, atol=1e-5) and np.isclose(desired_reciprocity, 1.0, atol=1e-5):
            # Both r and desired_reciprocity are approximately 1.0
            pass
        elif np.isclose(r, 1.0, atol=1e-5):
            non_zero_elements = adj_matrix[adj_matrix != 0]
            # Check if there are non-zero elements to avoid errors
            if non_zero_elements.size == 0:
                raise ValueError("Matrix contains no non-zero elements.")
            min_non_zero_value = np.min(non_zero_elements)
            perturbation = np.random.rand(*adj_matrix.shape) * min_non_zero_value #0.0001
            non_zero_mask = adj_matrix != 0
            perturbation_values = perturbation[non_zero_mask]
            mean_perturbation = np.mean(perturbation_values)
            # Adjust perturbation to have zero mean
            adjusted_perturbation = perturbation - mean_perturbation
            # Apply adjusted perturbation only to non-zero entries in adj_matrix
            adj_matrix[non_zero_mask] += adjusted_perturbation[non_zero_mask]
            # perturbation -= np.mean(perturbation)  # Adjust to have zero mean

            # adj_matrix[non_zero_mask] += perturbation[non_zero_mask]
            # adj_matrix = adj_matrix + perturbation
            adj_matrix = adjust_matrix(adj_matrix, L, desired_reciprocity)
        else:
            adj_matrix = adjust_matrix(adj_matrix, L, desired_reciprocity)
    else:
        adj_matrix = adjust_matrix_gradually(adj_matrix, L, desired_reciprocity, increment=0.01,
                                                 max_iter=num_iter, max_value=1000, min_value=0.01)
        hat_W1, hat_r_new = calculate_W1_and_r(adj_matrix, L)

    return adj_matrix


def convert_binary_to_weighted_matrix(X):
    return X * np.random.uniform(0.1, 1, X.shape)


def compute_density(adjmatrix):
    adjmatrix = adjmatrix.astype('bool')
    return adjmatrix.sum() / (adjmatrix.shape[0] * (adjmatrix.shape[0] - 1))

def compute_reciprocity_binary(adjmatrix):
    adjmatrix = adjmatrix.astype('bool')
    L = adjmatrix.sum()
    if L == 0:
        reciprocity = 0
    else:
        Rest = np.abs(adjmatrix ^ adjmatrix.T)
        Lsingle = 0.5*Rest.sum()
        reciprocity = np.float64(L-Lsingle) / L

    return reciprocity


def find_spectral_radius(adjacency_matrix):
    eigenvalues = np.linalg.eigvals(adjacency_matrix)
    spectral_radius = np.max(np.abs(eigenvalues))
    return spectral_radius


def in_and_out_degree(W):
    out_degree = np.sum(W, axis=1)
    in_degree = np.sum(W, axis=0)
    return out_degree, in_degree


def clustering_coefficient(W):
    g = convert_adjacency_matrix_to_graph(W)
    CC_avg = nx.average_clustering(g, nodes=None, weight="weight", count_zeros=True)
    return CC_avg


def compute_laplacian_matrix(w):
    return laplacian(w)


def generate_random_graph(num_nodes, density, rand_seed=42):
    np.random.seed(rand_seed)
    w = np.random.rand(num_nodes, num_nodes) < density
    np.fill_diagonal(w,0)
    return w


def generate_ER_graph(num_nodes, density, rand_seed=42):
    G = nx.fast_gnp_random_graph(num_nodes, density, seed=rand_seed, directed=True)
    w = convert_graph_to_adjacency_matrix(G)
    np.fill_diagonal(w, 0)
    return w


def generate_smallworld_graph(num_nodes, rewiring_prob=0.6, num_neighbors=10, rand_seed=42):
    w = nx.adjacency_matrix(nx.connected_watts_strogatz_graph(num_nodes, num_neighbors, rewiring_prob,
                                                               seed=rand_seed)).toarray()
    np.fill_diagonal(w, 0)
    return w


def generate_modular_graph(sz, pr, rand_seed=42):
    w = nx.adjacency_matrix(nx.stochastic_block_model(sz, pr, seed=rand_seed)).toarray()
    np.fill_diagonal(w, 0)
    return w

def random_weight_assignment(w, rand_seed=283):
    np.random.seed(rand_seed)
    p = np.random.rand(w.shape[0], w.shape[1])
    return np.multiply(w, p)


def FloydWarshall_Numba(adjmatrix, weighted_dist=False):
    @jit(nopython=True)
    def FW_Undirected(distmatrix):
        """The Floyd-Warshall algorithm for undirected networks
        """
        N = len(distmatrix)
        for k in range(N):
            for i in range(N):
                for j in range(i, N):
                    d = distmatrix[i, k] + distmatrix[k, j]
                    if distmatrix[i, j] > d:
                        distmatrix[i, j] = d
                        distmatrix[j, i] = d

    @jit(nopython=True)
    def FW_Directed(distmatrix):
        N = len(distmatrix)
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    d = distmatrix[i, k] + distmatrix[k, j]
                    if distmatrix[i, j] > d:
                        distmatrix[i, j] = d

    if weighted_dist:
        distmatrix = np.where(adjmatrix == 0, np.inf, adjmatrix)
    else:
        distmatrix = np.where(adjmatrix == 0, np.inf, 1)

    # 1.2) Find out whether the network is directed or undirected
    recip = compute_reciprocity_weighted(adjmatrix)
    if recip == 1.0:
        FW_Undirected(distmatrix)
    else:
        FW_Directed(distmatrix)

    return distmatrix


def compute_modularity_index(adj_mat):
    G = convert_adjacency_matrix_to_graph(adj_mat)
    partition = community_louvain.best_partition(G)
    modularity = community_louvain.modularity(partition, G)

    return modularity


def FloydWarshal_dir_weighted(w):
    INF = np.inf
    V = w.shape[0]
    dist = w.copy()
    dist = np.where(w == 0, INF, w)
    for k in range(V):
        # Use broadcasting to calculate the shortest path from i to j via k
        dist = np.minimum(dist, dist[:, k].reshape(-1, 1) + dist[k, :])

    return dist

def weight_to_length(weight_matrix):
    weight_matrix = np.array(weight_matrix, dtype=float)
    max_weight = np.max(weight_matrix)
    if max_weight == 0:
        raise ValueError("Maximum weight is zero, cannot perform weight-to-length remapping.")
    normalized_matrix = weight_matrix / max_weight
    inverse_matrix = np.copy(normalized_matrix)
    print('min and max: ', inverse_matrix.min(), inverse_matrix.max())
    non_zero_mask = normalized_matrix != 0
    inverse_matrix[non_zero_mask] = 1 / (normalized_matrix[non_zero_mask] + 1)
    print('min and max: ', inverse_matrix.min(), inverse_matrix.max())
    # length_matrix = np.log10((weight_matrix / max_weight) + 1)
    return inverse_matrix

def BellmanFord(w):
    w = weight_to_length(w)
    # if np.any(w < 0):
    #     raise ValueError("Weight matrix contains negative weights.")
    G = convert_adjacency_matrix_to_graph(w)
    try:
        length = dict(nx.all_pairs_bellman_ford_path_length(G, weight='weight'))
        dist = np.zeros(w.shape)
        np.fill_diagonal(dist, 0)
        dist[w == 0] = np.inf
        for source in range(w.shape[0]):
            for target in range(w.shape[0]):
                try:
                    # Attempt to retrieve the length value
                    dist[source, target] = length[source][target]
                except KeyError:
                    # If the key is not found, assign np.inf or np.nan
                    dist[source, target] = np.inf
    except nx.NetworkXUnbounded:
        # Handle the case where a negative weight cycle is detected
        print("Negative weight cycle detected. Returning a zero matrix.")
        # Return a zero matrix in case of negative weight cycle
        dist = np.zeros(w.shape)

    return dist


def departure_from_normality(M):
    M = np.array(M)
    # Calculate the Frobenius norm of M
    norm_F = norm(M, 'fro')
    # Compute the eigenvalues of M
    eigenvalues = np.linalg.eigvals(M)
    # Compute the sum of the squares of the eigenvalues
    sum_squares_eigenvalues = np.sum(np.abs(eigenvalues)**2)
    # Compute the departure from normality
    d_F = np.sqrt(norm_F**2 - sum_squares_eigenvalues)
    return d_F/norm_F


