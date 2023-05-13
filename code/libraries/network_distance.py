import numpy as np
import networkx as nx
from pyemd import emd as _emd
from scipy import spatial
from scipy.sparse import csgraph
from multiprocessing import Pool, Manager

def calculate_Q(network, edge_weights = None):
   A = nx.adjacency_matrix(network, weight = edge_weights).todense().astype(float)
   return np.linalg.pinv(csgraph.laplacian(np.matrix(A), normed = False))

def ge(src, trg, network, edge_weights = None, Q = None, normed = True):
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   src = np.array([src[n] if n in src else 0. for n in network.nodes()])
   trg = np.array([trg[n] if n in trg else 0. for n in network.nodes()])
   if normed:
      src = src / src.sum()
      trg = trg / trg.sum()
   diff = src - trg
   if Q is None:
      Q = calculate_Q(network, edge_weights = edge_weights)
   return np.sqrt(diff.T.dot(np.array(Q).dot(diff)))

def calculate_v(network, edge_weights = None):
   A = nx.adjacency_matrix(network, weight = edge_weights).todense().astype(float)
   laplacian = csgraph.laplacian(np.matrix(A), normed = False)
   l, v = np.linalg.eig(laplacian)
   idx = l.argsort()
   l = l[idx]
   v = np.diag(l).dot(v[:,idx])
   return v

def spectrum(src, trg, network, edge_weights = None, v = None, normed = True):
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   src = np.array([src[n] if n in src else 0. for n in network.nodes()])
   trg = np.array([trg[n] if n in trg else 0. for n in network.nodes()])
   if normed:
      src = src / src.sum()
      trg = trg / trg.sum()
   if v is None:
      v = calculate_v(network, edge_weights = edge_weights)
   src = src.dot(v)
   trg = trg.dot(v)
   return spatial.distance.euclidean(src, trg)

def _spl(x):
   x[2][x[1]] = dict(nx.shortest_path_length(x[0], source = x[1]))

def calculate_spl(network, nonzero_nodes, n_proc, return_as_dict = False):
   manager = Manager()
   shortest_path_lengths = manager.dict()
   pool = Pool(processes = n_proc)
   _ = pool.map(_spl, [(network, n, shortest_path_lengths) for n in nonzero_nodes])
   pool.close()
   pool.join()
   shortest_path_lengths = dict(shortest_path_lengths)
   if return_as_dict:
      return shortest_path_lengths
   else:
      return np.array([[shortest_path_lengths[i][j] for i in nonzero_nodes] for j in nonzero_nodes], dtype = "float64")

def emd(src, trg, network, edge_weights = None, shortest_path_lengths = None, n_proc = 1, normed = True):
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   nonzero_nodes = list(set(src) | set(trg))
   src = np.array([src[n] if n in src else 0. for n in nonzero_nodes], dtype = float)
   trg = np.array([trg[n] if n in trg else 0. for n in nonzero_nodes], dtype = float)
   if normed:
      src = src / src.sum()
      trg = trg / trg.sum()
   if shortest_path_lengths is None:
      shortest_path_lengths = calculate_spl(network, nonzero_nodes, n_proc)
   else:
      shortest_path_lengths = np.array([[shortest_path_lengths[i][j] for i in nonzero_nodes] for j in nonzero_nodes], dtype = "float64")
   return _emd(src, trg, shortest_path_lengths)
