# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from scipy.spatial import distance_matrix
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from rdkit import Chem
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch_cluster import radius_graph
from typing import Tuple
from torch_geometric.typing import SparseTensor

def floyd_warshall(A):
    n = A.shape[0]
    D = np.zeros((n,n), dtype=np.int16)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                pass
            elif A[i,j] == 0:
                D[i,j] = 510
            else:
                D[i,j] = 1
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                old_dist = D[i,j]
                new_dist = D[i,k] + D[k,j]
                if new_dist < old_dist:
                    D[i,j] = new_dist
    return D

class DistanceDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pos = self.dataset[idx].view(-1, 3).numpy()
        dist = distance_matrix(pos, pos).astype(np.float32)
        return torch.from_numpy(dist)


class EdgeTypeDataset(BaseWrapperDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, num_types: int):
        self.dataset = dataset
        self.num_types = num_types

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        node_input = self.dataset[index].clone()
        offset = node_input.view(-1, 1) * self.num_types + node_input.view(1, -1)
        return offset

class TGTEdgeDataset(BaseWrapperDataset):
    def __init__(self, tgtnode_dataset: torch.utils.data.Dataset, coord_dataset=None, bond_type='hole'):
        self.tgtnode_dataset = tgtnode_dataset
        self.bond_type = bond_type
        self.coord_dataset = coord_dataset
        
    def get_bond(self, smi):
        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        
        return edge_index, edge_attr
            
    @lru_cache(maxsize=16)
    def __getitem__(self, index: int, coord_dataset=None):
        smiles = self.tgtnode_dataset[index]['smi']
        edges, edge_features = self.get_bond(smiles)
        num_nodes = self.tgtnode_dataset[index]['num_nodes']
            
        if self.bond_type == 'hole':
            node_mask = self.tgtnode_dataset[index]['node_mask']
            edge_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)   # old version, tgt-training_schemes-pretrain, global-graph
        elif self.bond_type == 'bond':
            edge_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.int32)
            edge_mask[edges[0], edges[1]] = 1
        elif self.bond_type == 'radius':
            coord = self.coord_dataset[index]
            radius_index = radius_graph(coord, r=5.0, loop=False, 
                max_num_neighbors=1000)
            edge_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.int32)
            edge_mask[radius_index[0], radius_index[1]] = 1
        
        return {'edges':torch.from_numpy(edges), 'edge_features':torch.from_numpy(edge_features), 'edge_mask':edge_mask}

class DistMatDataset(BaseWrapperDataset):
    def __init__(self, tgtnode_dataset: torch.utils.data.Dataset, tgtedge_dataset: torch.utils.data.Dataset, coord_dataset):
        self.tgtnode_dataset = tgtnode_dataset
        self.tgtedge_dataset = tgtedge_dataset
        self.coord_dataset = coord_dataset
    
    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        num_nodes = self.tgtnode_dataset[index]['num_nodes'].clone()
        edges = self.tgtedge_dataset[index]['edges'].clone()
        edge_feats = self.tgtedge_dataset[index]['edge_features'].clone()
        
        A = np.zeros((num_nodes,num_nodes),dtype=np.int16)
        feature_matrix = np.zeros((num_nodes,num_nodes,edge_feats.shape[-1]),dtype=np.int16)
        for k in range(edges.shape[0]):
            i,j = edges[k,0], edges[k,1]
            A[i,j] = 1
            feature_matrix[i,j] = edge_feats[k]
            
        # distance_matrix = torch.from_numpy(floyd_warshall(A))
        coords = self.coord_dataset[index]
        distance_matrix = torch.norm(coords.unsqueeze(0) - coords.unsqueeze(1), dim=-1)

        return {'distance_matrix':distance_matrix, 'feature_matrix': torch.from_numpy(feature_matrix)}

class TGTNodeDataset(BaseWrapperDataset):
    def __init__(self, atom_dataset: torch.utils.data.Dataset):
        self.atom_dataset = atom_dataset
 
    def get_node(self, mol):
        
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        return x
        
    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        smi = self.atom_dataset[index]
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        num_nodes = mol.GetNumAtoms()
        node_mask = np.ones(num_nodes, dtype=np.uint8)
        node_features = self.get_node(mol)
        return {'smi':smi,
                'num_nodes':torch.tensor(num_nodes),
                'node_mask':torch.from_numpy(node_mask),
                'node_features':torch.from_numpy(node_features)
                }
        
class TGTtriDataset(BaseWrapperDataset):
    def __init__(self, edge_dataset: torch.utils.data.Dataset, coord_dataset: torch.utils.data.Dataset, node_dataset: torch.utils.data.Dataset):
        self.edge_dataset = edge_dataset
        self.coord_dataset = coord_dataset
        self.node_dataset = node_dataset
    
    def triplets(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            row, col = edge_index  # j->i

            value = torch.arange(row.size(0), device=row.device)    # [0-row]
            adj_t = SparseTensor(row=col, col=row, value=value,
                                sparse_sizes=(num_nodes, num_nodes))
            adj_t_row = adj_t[row]
            num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)  # tri_num on each node

            # Node indices (k->j->i) for triplets.
            idx_i = col.repeat_interleave(num_triplets)
            idx_j = row.repeat_interleave(num_triplets)
            idx_k = adj_t_row.storage.col()
            mask = idx_i != idx_k  # Remove i == k triplets.
            idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

            # Edge indices (k-j, j->i) for triplets.
            idx_kj = adj_t_row.storage.value()[mask]
            idx_ji = adj_t_row.storage.row()[mask]

            return idx_i, idx_j, idx_k, idx_kj, idx_ji

    
    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        bond_edges = self.edge_dataset[index]['edges']
        coord = self.coord_dataset[index]
        num_nodes = self.node_dataset[index]['num_nodes']
        i, j = bond_edges
        R_ij = torch.norm(coord[i] - coord[j], p=2, dim=-1)
        # threebody_mask =  R_ij < R_ij.mean().item()
        tri_i, tri_j, tri_k, tri_kj, tri_ji = self.triplets(bond_edges.long(), num_nodes=num_nodes)
        # tri_edges = np.array([
        #     [bond_edges[:, tri_i], bond_edges[:, tri_j], bond_edges[:, tri_k]]  # 每个三体交互的三条边
        # ]).transpose(2, 0, 1)
        
        tri_matrix = np.zeros((num_nodes, num_nodes, num_nodes))
        for idx in range(len(tri_i)):
            node_i = bond_edges[0, tri_i[idx]]  
            node_j = bond_edges[1, tri_j[idx]]  
            node_k = bond_edges[1, tri_k[idx]]  
            
            tri_matrix[node_i, node_j, node_k] = 1
            
        return torch.tensor(tri_matrix)