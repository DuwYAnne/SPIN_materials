import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, InMemoryDataset, Dataset
class GraphConstructor():
    def __init__(self, cutoff=6.0, tol=1e-8, max_neighbors=12):
        self.cutoff=cutoff
        self.tol=tol
        self.max_neighbors=max_neighbors
    def structure_to_graph(self, crystal, target, sid=None):
        pos = np.array(crystal.cart_coords)
        lattice = np.array(crystal.lattice.matrix)
        atomic_numbers = np.array([x.number for x in crystal.species])
        natoms = pos.shape[0]
        c_idx, n_idx, cell_offsets, distance = find_points_in_spheres(
            all_coords=pos,
            center_coords=pos,
            lattice=lattice,
            r=self.cutoff,
            tol=self.tol,
            pbc=np.array([1,1,1], dtype=int)
        )
        pos = torch.Tensor(pos)
        lattice = torch.Tensor(lattice)
        atomic_numbers=torch.Tensor(atomic_numbers)
        noself = (c_idx != n_idx) | (distance > self.tol)
        c_idx = c_idx[noself]
        n_idx = n_idx[noself]
        distance = distance[noself]
        cell_offsets = cell_offsets[noself]
        r_idx = []
        for i in range(len(crystal)):
            idx_i = (c_idx == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(distance[idx_i])[: self.max_neighbors]
            r_idx.append(idx_i[idx_sorted])
        r_idx = np.concatenate(r_idx)
        edge_index = torch.LongTensor(np.vstack([c_idx[r_idx], n_idx[r_idx]])) # has shape(2, m)
        cell_offsets = torch.LongTensor(cell_offsets[r_idx])
        distance = torch.FloatTensor(distance[r_idx])
        return Data(
                    cell=lattice.view(1, 3, 3),
                    pos=pos,
                    atomic_numbers=atomic_numbers,
                    natoms=natoms,
                    edge_index=edge_index,
                    cell_offsets=cell_offsets,
                    distances=distance,
                    y=target,
                    sid=sid if sid is not None else None
        )


class GaussianDistance:
    '''
    Helper class for computing a gaussian filter over a distance matrix.

    args:
        centers: np.ndarray of shape (n, m)
        width: float
    
    returns:
        np.ndarray of shape (n, m) where each distance vector is passed over gaussian filter.
    '''
    def __init__(self, centers:np.ndarray, width:float):
        self.centers = centers
        self.width=width
    def __call__(self, x: np.ndarray):
        x = np.array(x)
        return np.exp(-((x[:, None] - self.centers[None, :])**2 / self.width**2))

class StructData(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
    
