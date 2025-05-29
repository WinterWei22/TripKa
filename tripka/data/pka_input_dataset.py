# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import collections
import torch
from itertools import chain
from unicore.data.data_utils import collate_tokens, collate_tokens_2d  
from .coord_pad_dataset import collate_tokens_coords
def get_probMat(edge_matrix, num_nodes,default_value=0.15):
    """
    edge_matrix: [b,i,j], 统计后的edge_matrix
    num_nodes: [b]
    return: prob_matrix
    """
    prob_matrix = torch.zeros(edge_matrix.shape).to(edge_matrix.device)
    for i in range(edge_matrix.shape[0]):
        submatrix = edge_matrix[i, :num_nodes[i], :num_nodes[i]].clone()
        # 0->default value, others->
        temp_matrix = torch.where(submatrix == 0, 
                                  torch.tensor(default_value), 
                                  torch.min(submatrix * 0.2, torch.tensor(1.0)))
        
        prob_matrix[i, :num_nodes[i], :num_nodes[i]] = temp_matrix

    return prob_matrix

def trimat_to_mask(triplet_matrix, num_nodes):
    """
    triplet_matrix: [b,i,j,k]
    num_nodes: [b]
    return: prob_matrix
    """
    edge_matrix = triplet_matrix.sum(-1)
    prob_matrix = get_probMat(edge_matrix, num_nodes)
    
    return prob_matrix

def collate_tokens_2d_2(
    tensor_list,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    max_n = max(tensor.size(0) for tensor in tensor_list)
    
    # 计算需要扩展到的大小
    new_size = ((max_n + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    
    # 用0填充并且将所有tensor堆叠成一个三维张量
    padded_tensors = []
    for tensor in tensor_list:
        pad_size = (0, 0, 0, new_size - tensor.size(0))
        padded_tensor = torch.nn.functional.pad(tensor, pad_size, mode='constant', value=0)
        padded_tensors.append(padded_tensor)
    
    # 将填充后的tensor堆叠在一起
    result = torch.stack(padded_tensors, dim=0)
    
    return result

def stack_tensor_3d(tensor_list, pad_to_multiple):
    assert len(tensor_list[0].shape) == 3 and tensor_list[0].shape[0] == tensor_list[0].shape[1]
    max_n = max(tensor.size(0) for tensor in tensor_list)
    
    # 计算需要扩展到的大小
    new_size = ((max_n + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    
    # 用0填充并且将所有tensor堆叠成一个四维张量
    padded_tensors = []
    for tensor in tensor_list:
        pad_size = (0, 0, 0, new_size - tensor.size(1), 0, new_size - tensor.size(0))
        padded_tensor = torch.nn.functional.pad(tensor, pad_size, mode='constant', value=0)
        padded_tensors.append(padded_tensor)
    
    # 将填充后的tensor堆叠在一起
    result = torch.stack(padded_tensors, dim=0)
    
    return result

def stack_tensor_4d(tensor_list, pad_to_multiple):
    # Step 1: 找出list中最大num_nodes
    num_max_nodes = max([tensor.shape[0] for tensor in tensor_list])
    
    # Step 2: 将num_max_nodes向上取整到8的倍数
    num_max_nodes = ((num_max_nodes + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    
    # Step 3: 初始化一个四维张量 [batch_size, num_max_nodes, num_max_nodes, num_max_nodes]
    batch_size = len(tensor_list)
    padded_batch_tensor = torch.zeros((batch_size, num_max_nodes, num_max_nodes, num_max_nodes))
    
    # Step 4: 对每个tensor进行padding并填充到padded_batch_tensor中
    for idx, tensor in enumerate(tensor_list):
        num_nodes = tensor.shape[0]
        padded_batch_tensor[idx, :num_nodes, :num_nodes, :num_nodes] = tensor
    
    return padded_batch_tensor

class PKAInputDataset(BaseWrapperDataset):
    
    def __init__(self, idx2key, src_tokens, src_charges, src_coord, src_distance, src_edge_type, token_pad_idx, charge_pad_idx, split='train', conf_size=10):
        
        self.idx2key = idx2key
        self.dataset = src_tokens
        self.src_tokens = src_tokens
        self.src_charges = src_charges
        self.src_coord = src_coord
        self.src_distance = src_distance
        self.src_edge_type = src_edge_type
        self.token_pad_idx = token_pad_idx
        self.charge_pad_idx = charge_pad_idx
        self.split = split
        self.conf_size = conf_size
        self.left_pad = False
        self._init_rec2mol()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def _init_rec2mol(self):
        self.rec2mol = collections.defaultdict(list)
        if self.split in ['train','train.small']:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                self.rec2mol[smi_idx].append(i)
        else:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                self.rec2mol[smi_idx].extend([i * self.conf_size + j for j in range(self.conf_size)])


    def __len__(self):
        return len(self.rec2mol)

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        # 获取数据
        mol_list = self.rec2mol[index]
        src_tokens_list = []
        src_charges_list = []
        src_coord_list = []
        src_distance_list = []
        src_edge_type_list = []
        for i in mol_list:
            src_tokens_list.append(self.src_tokens[i])
            src_charges_list.append(self.src_charges[i])
            src_coord_list.append(self.src_coord[i])
            src_distance_list.append(self.src_distance[i])
            src_edge_type_list.append(self.src_edge_type[i])

        return src_tokens_list, src_charges_list,src_coord_list,src_distance_list,src_edge_type_list
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
    
    def collater(self, samples):
        batch = [len(samples[i][0]) for i in range(len(samples))]
        # collate_tokens 是将list转化为tensor，并且补齐的函数
        src_tokens, src_charges, src_coord, src_distance, src_edge_type = [list(chain.from_iterable(i)) for i in zip(*samples)]
        src_tokens = collate_tokens(src_tokens, self.token_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_charges = collate_tokens(src_charges, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_coord = collate_tokens_coords(src_coord, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_distance = collate_tokens_2d(src_distance, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_edge_type = collate_tokens_2d(src_edge_type, 0, left_pad=self.left_pad, pad_to_multiple=8)

        return src_tokens, src_charges, src_coord, src_distance, src_edge_type, batch

class PKAMLMInputDataset(BaseWrapperDataset):
    def __init__(self, idx2key, src_tokens, src_charges, src_coord, src_distance, src_edge_type, token_targets, charge_targets, dist_targets, coord_targets, token_pad_idx, charge_pad_idx, split='train', conf_size=10):
        self.idx2key = idx2key
        self.dataset = src_tokens
        self.src_tokens = src_tokens
        self.src_charges = src_charges
        self.src_coord = src_coord
        self.src_distance = src_distance
        self.src_edge_type = src_edge_type
        self.token_targets = token_targets
        self.charge_targets = charge_targets
        self.dist_targets = dist_targets
        self.coord_targets = coord_targets
        self.token_pad_idx = token_pad_idx
        self.charge_pad_idx = charge_pad_idx
        self.split = split
        self.conf_size = conf_size
        self.left_pad = False
        self._init_rec2mol()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def _init_rec2mol(self):
        self.rec2mol = collections.defaultdict(list)
        if self.split in ['train','train.small']:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                self.rec2mol[smi_idx].append(i)
        else:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                self.rec2mol[smi_idx].extend([i * self.conf_size + j for j in range(self.conf_size)])


    def __len__(self):
        return len(self.rec2mol)

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        mol_list = self.rec2mol[index]
        src_tokens_list = []
        src_charges_list = []
        src_coord_list = []
        src_distance_list = []
        src_edge_type_list = []
        token_targets_list = []
        charge_targets_list = []
        coord_targets_list = []
        dist_targets_list = []
        for i in mol_list:
            src_tokens_list.append(self.src_tokens[i])
            src_charges_list.append(self.src_charges[i])
            src_coord_list.append(self.src_coord[i])
            src_distance_list.append(self.src_distance[i])
            src_edge_type_list.append(self.src_edge_type[i])
            token_targets_list.append(self.token_targets[i])
            charge_targets_list.append(self.charge_targets[i])
            coord_targets_list.append(self.coord_targets[i])
            dist_targets_list.append(self.dist_targets[i])

        return src_tokens_list, src_charges_list,src_coord_list,src_distance_list,src_edge_type_list, token_targets_list, charge_targets_list, coord_targets_list, dist_targets_list
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
    
    def collater(self, samples):
        batch = [len(samples[i][0]) for i in range(len(samples))]

        src_tokens, src_charges, src_coord, src_distance, src_edge_type, token_targets, charge_targets, coord_targets, dist_targets  = [list(chain.from_iterable(i)) for i in zip(*samples)]
        src_tokens = collate_tokens(src_tokens, self.token_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_charges = collate_tokens(src_charges, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_coord = collate_tokens_coords(src_coord, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_distance = collate_tokens_2d(src_distance, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_edge_type = collate_tokens_2d(src_edge_type, 0, left_pad=self.left_pad, pad_to_multiple=8)
        token_targets = collate_tokens(token_targets, self.token_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        charge_targets = collate_tokens(charge_targets, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        coord_targets = collate_tokens_coords(coord_targets, 0, left_pad=self.left_pad, pad_to_multiple=8)
        dist_targets = collate_tokens_2d(dist_targets, 0, left_pad=self.left_pad, pad_to_multiple=8)

        return src_tokens, src_charges, src_coord, src_distance, src_edge_type, batch, charge_targets, coord_targets, dist_targets, token_targets

class TGT_PKAMLMInputDataset(BaseWrapperDataset):
    def __init__(self, idx2key, src_tokens, src_charges, src_coord, src_distance, src_edge_type, token_targets, charge_targets, dist_targets, coord_targets, token_pad_idx, charge_pad_idx, 
                 tgtnode_dataset, tgtedge_dataset, tgtdist_dataset, triplet_dataset, split='train', conf_size=10, bond_triplet=False):
        
        self.idx2key = idx2key
        self.dataset = src_tokens
        self.src_tokens = src_tokens
        self.src_charges = src_charges
        self.src_coord = src_coord
        self.src_distance = src_distance
        self.src_edge_type = src_edge_type
        self.token_targets = token_targets
        self.charge_targets = charge_targets
        self.dist_targets = dist_targets
        self.coord_targets = coord_targets
        self.token_pad_idx = token_pad_idx
        self.charge_pad_idx = charge_pad_idx
        self.tgtnode_dataset = tgtnode_dataset
        self.tgtedge_dataset = tgtedge_dataset
        self.tgtdist_dataset = tgtdist_dataset
        
        self.triplet_dataset = triplet_dataset
        
        self.split = split
        self.conf_size = conf_size
        self.left_pad = False
        self.bond_triplet = bond_triplet
        self._init_rec2mol()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def _init_rec2mol(self):
        self.rec2mol = collections.defaultdict(list)
        if self.split in ['train','train.small']:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                self.rec2mol[smi_idx].append(i)
        else:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                # self.rec2mol[smi_idx].append(i)
                self.rec2mol[smi_idx].extend([i * self.conf_size + j for j in range(self.conf_size)])


    def __len__(self):
        return len(self.rec2mol)

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        mol_list = self.rec2mol[index]
        src_tokens_list = []
        src_charges_list = []
        src_coord_list = []
        src_distance_list = []
        src_edge_type_list = []
        token_targets_list = []
        charge_targets_list = []
        coord_targets_list = []
        dist_targets_list = []
        # tgt features
        # nodes
        num_nodes_list = []
        node_features_list = []
        node_mask_list = []
        # edges
        edge_mask_list = []
        # matrix
        distance_matrix_list = []
        feature_matrix_list = []
        
        triplet_matrix_list = []
        for i in mol_list:
            src_tokens_list.append(self.src_tokens[i])
            src_charges_list.append(self.src_charges[i])
            src_coord_list.append(self.src_coord[i])
            src_distance_list.append(self.src_distance[i])
            src_edge_type_list.append(self.src_edge_type[i])
            token_targets_list.append(self.token_targets[i])
            charge_targets_list.append(self.charge_targets[i])
            coord_targets_list.append(self.coord_targets[i])
            dist_targets_list.append(self.dist_targets[i])
            
            num_nodes_list.append(self.tgtnode_dataset[i]['num_nodes'])
            node_features_list.append(self.tgtnode_dataset[i]['node_features'])
            node_mask_list.append(self.tgtnode_dataset[i]['node_mask'])
            edge_mask_list.append(self.tgtedge_dataset[i]['edge_mask'])
            distance_matrix_list.append(self.tgtdist_dataset[i]['distance_matrix'])
            feature_matrix_list.append(self.tgtdist_dataset[i]['feature_matrix'])

            if self.bond_triplet:
                triplet_matrix_list.append(self.triplet_dataset[i])

        return src_tokens_list, src_charges_list,src_coord_list,src_distance_list,src_edge_type_list, token_targets_list, charge_targets_list, \
            coord_targets_list, dist_targets_list, num_nodes_list, node_features_list, node_mask_list, edge_mask_list, distance_matrix_list, feature_matrix_list, \
            triplet_matrix_list
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
    
    def collater(self, samples):
        batch = [len(samples[i][0]) for i in range(len(samples))]

        src_tokens, src_charges, src_coord, src_distance, src_edge_type, token_targets, charge_targets, coord_targets, dist_targets, \
            num_nodes, node_features, node_mask, edge_mask, distance_matrix, feature_matrix, triplet_matrix = [list(chain.from_iterable(i)) for i in zip(*samples)]
        src_tokens = collate_tokens(src_tokens, self.token_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_charges = collate_tokens(src_charges, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_coord = collate_tokens_coords(src_coord, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_distance = collate_tokens_2d(src_distance, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_edge_type = collate_tokens_2d(src_edge_type, 0, left_pad=self.left_pad, pad_to_multiple=8)
        token_targets = collate_tokens(token_targets, self.token_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        charge_targets = collate_tokens(charge_targets, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        coord_targets = collate_tokens_coords(coord_targets, 0, left_pad=self.left_pad, pad_to_multiple=8)
        dist_targets = collate_tokens_2d(dist_targets, 0, left_pad=self.left_pad, pad_to_multiple=8)
        num_nodes = torch.stack(num_nodes)
        node_features = collate_tokens_2d_2(node_features, 0, left_pad=self.left_pad, pad_to_multiple=8)
        node_mask = collate_tokens(node_mask, 0, left_pad=self.left_pad, pad_to_multiple=8)
        edge_mask = collate_tokens_2d(edge_mask, 0, left_pad=self.left_pad, pad_to_multiple=8)
        distance_matrix = collate_tokens_2d(distance_matrix, 0, left_pad=self.left_pad, pad_to_multiple=8)
        feature_matrix = stack_tensor_3d(feature_matrix, pad_to_multiple=8)
        if self.bond_triplet: 
            triplet_matrix = stack_tensor_4d(triplet_matrix, pad_to_multiple=8)
            triplet_matrix = trimat_to_mask(triplet_matrix, num_nodes)
        else:
            triplet_matrix = None
        # return {'num_nodes': num_nodes, 'node_mask': node_mask, 'rdkit_coords': src_coord, 'node_features': node_features,'distance_matrix': distance_matrix, 
        #         'feature_matrix':feature_matrix, 'edge_mask': edge_mask, 'src_charges': src_charges, 'src_tokens': src_tokens, 'dist_input': src_distance,
        #         'src_tokens':src_tokens ,'src_edge_type':src_edge_type, 'token_targets': token_targets ,'charge_targets': charge_targets, 'coord_targets': coord_targets,
        #         'dist_targets':dist_targets}

        return  src_tokens, src_charges, src_coord, src_distance, src_edge_type, \
                num_nodes, node_features, node_mask, edge_mask, distance_matrix, feature_matrix, triplet_matrix, \
                batch, charge_targets, coord_targets, dist_targets, token_targets,

class TGT_PKAInputDataset(BaseWrapperDataset):
    def __init__(self, idx2key, src_tokens, src_charges, src_coord, src_distance, src_edge_type, token_pad_idx, charge_pad_idx, 
                 tgtnode_dataset, tgtedge_dataset, tgtdist_dataset, triplet_dataset ,split='train', conf_size=10, bond_triplet=False, qm_dataset=None, 
                 wiberg_bonds=None, mulliken_charges=None, wiberg_thredhold=0.001):
        
        self.idx2key = idx2key
        self.dataset = src_tokens
        self.src_tokens = src_tokens
        self.src_charges = src_charges
        self.src_coord = src_coord
        self.src_distance = src_distance
        self.src_edge_type = src_edge_type
        
        self.token_pad_idx = token_pad_idx
        self.charge_pad_idx = charge_pad_idx
        
        self.split = split
        self.conf_size = conf_size
        self.left_pad = False
        self.bond_triplet = bond_triplet
        self._init_rec2mol()
        self.set_epoch(None)
        
        self.tgtnode_dataset = tgtnode_dataset
        self.tgtedge_dataset = tgtedge_dataset
        self.tgtdist_dataset = tgtdist_dataset

        self.triplet_dataset = triplet_dataset

        self.qm_dataset = qm_dataset
        self.wiberg_bonds = wiberg_bonds
        self.mulliken_charges = mulliken_charges
        self.wiberg_thredhold = wiberg_thredhold
        
    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def _init_rec2mol(self):
        self.rec2mol = collections.defaultdict(list)
        if self.split in ['train','train.small']:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                self.rec2mol[smi_idx].append(i)
        else:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                # self.rec2mol[smi_idx].append(i)
                self.rec2mol[smi_idx].extend([i * self.conf_size + j for j in range(self.conf_size)])

    def __len__(self):
        return len(self.rec2mol)

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        mol_list = self.rec2mol[index]
        src_tokens_list = []
        src_charges_list = []
        src_coord_list = []
        src_distance_list = []
        src_edge_type_list = []
        # tgt features
        # nodes
        num_nodes_list = []
        node_features_list = []
        node_mask_list = []
        # edges
        edge_mask_list = []
        # matrix
        distance_matrix_list = []
        feature_matrix_list = []
        
        # triplet
        triplet_matrix_list = []
        qm_features_list = []

        # psi4 qm targets
        wiberg_bonds_list = []
        mulliken_charges_list = []

        for i in mol_list:
            src_tokens_list.append(self.src_tokens[i])
            src_charges_list.append(self.src_charges[i])
            src_coord_list.append(self.src_coord[i])
            src_distance_list.append(self.src_distance[i])
            src_edge_type_list.append(self.src_edge_type[i])
            if self.qm_dataset != None:
                qm_features_list.append(self.qm_dataset[i])
            if self.wiberg_bonds != None:
                wiberg_bonds_list.append(self.wiberg_bonds[i])
            if self.mulliken_charges != None:
                mulliken_charges_list.append(self.mulliken_charges[i])

            num_nodes_list.append(self.tgtnode_dataset[i]['num_nodes'])
            node_features_list.append(self.tgtnode_dataset[i]['node_features'])
            node_mask_list.append(self.tgtnode_dataset[i]['node_mask'])
            edge_mask_list.append(self.tgtedge_dataset[i]['edge_mask'])
            distance_matrix_list.append(self.tgtdist_dataset[i]['distance_matrix'])
            feature_matrix_list.append(self.tgtdist_dataset[i]['feature_matrix'])


            if self.bond_triplet:
                triplet_matrix_list.append(self.triplet_dataset[i])

        return src_tokens_list, src_charges_list,src_coord_list,src_distance_list,src_edge_type_list, \
                num_nodes_list, node_features_list, node_mask_list, edge_mask_list, distance_matrix_list, feature_matrix_list, \
                triplet_matrix_list, qm_features_list, wiberg_bonds_list, mulliken_charges_list
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
    
    def collater(self, samples):
        batch = [len(samples[i][0]) for i in range(len(samples))]

        src_tokens, src_charges, src_coord, src_distance, src_edge_type, \
            num_nodes, node_features, node_mask, edge_mask, distance_matrix, feature_matrix, triplet_matrix, qm_features_list, \
            wiberg_bonds_list, mulliken_charges_list = [list(chain.from_iterable(i)) for i in zip(*samples)]
        
        src_tokens = collate_tokens(src_tokens, self.token_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_charges = collate_tokens(src_charges, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_coord = collate_tokens_coords(src_coord, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_distance = collate_tokens_2d(src_distance, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_edge_type = collate_tokens_2d(src_edge_type, 0, left_pad=self.left_pad, pad_to_multiple=8)
        num_nodes = torch.stack(num_nodes)
        node_features = collate_tokens_2d_2(node_features, 0, left_pad=self.left_pad, pad_to_multiple=8)
        node_mask = collate_tokens(node_mask, 0, left_pad=self.left_pad, pad_to_multiple=8)
        edge_mask = collate_tokens_2d(edge_mask, 0, left_pad=self.left_pad, pad_to_multiple=8)
        distance_matrix = collate_tokens_2d(distance_matrix, 0, left_pad=self.left_pad, pad_to_multiple=8)
        feature_matrix = stack_tensor_3d(feature_matrix, pad_to_multiple=8)

        if self.wiberg_bonds != None:
            wiberg_bonds = collate_tokens_2d(wiberg_bonds_list, 0, left_pad=self.left_pad, pad_to_multiple=8)
            output_tensor = torch.full_like(wiberg_bonds, -100.0)
            if self.wiberg_thredhold >= 1:
                """分bins多分类"""
                output_tensor[(wiberg_bonds > 0.1) & (wiberg_bonds <= 1.0)] = 0
                output_tensor[(wiberg_bonds > 1.0) & (wiberg_bonds <= self.wiberg_thredhold)] = 1
                output_tensor[(wiberg_bonds > self.wiberg_thredhold)] = 2
            else:
                """二分类"""
                output_tensor[(wiberg_bonds > 0) & (wiberg_bonds <= self.wiberg_thredhold)] = 1
                output_tensor[(wiberg_bonds > self.wiberg_thredhold)] = 0
            wiberg_bonds = output_tensor
        else:
            wiberg_bonds = None
        if self.mulliken_charges != None:
            mulliken_charges = collate_tokens(mulliken_charges_list, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        else:
            mulliken_charges = None
            
        if self.qm_dataset != None:
            qm_features = torch.stack(qm_features_list)
        else:
            qm_features = None

        if self.bond_triplet: 
            triplet_matrix = stack_tensor_4d(triplet_matrix, pad_to_multiple=8)
            triplet_matrix = trimat_to_mask(triplet_matrix, num_nodes)
            # print(triplet_matrix.shape)
        else:
            triplet_matrix = None
        

        # return {'num_nodes': num_nodes, 'node_mask': node_mask, 'rdkit_coords': src_coord, 'node_features': node_features,'distance_matrix': distance_matrix, 
        #         'feature_matrix':feature_matrix, 'edge_mask': edge_mask, 'src_charges': src_charges, 'src_tokens': src_tokens, 'dist_input': src_distance,
        #         'src_tokens':src_tokens ,'src_edge_type':src_edge_type, 'token_targets': token_targets ,'charge_targets': charge_targets, 'coord_targets': coord_targets,
        #         'dist_targets':dist_targets}

        return  src_tokens, src_charges, src_coord, src_distance, src_edge_type, \
                num_nodes, node_features, node_mask, edge_mask, distance_matrix, feature_matrix, triplet_matrix, qm_features, \
                wiberg_bonds, mulliken_charges, batch

class TGT_LOGDInputDataset(BaseWrapperDataset):
    def __init__(self, idx2key, src_tokens, src_charges, src_coord, src_distance, src_edge_type, token_pad_idx, charge_pad_idx, 
                 tgtnode_dataset, tgtedge_dataset, tgtdist_dataset, triplet_dataset ,split='train', conf_size=10, bond_triplet=False, qm_dataset=None):
        
        self.idx2key = idx2key
        self.dataset = src_tokens
        self.src_tokens = src_tokens
        self.src_charges = src_charges
        self.src_coord = src_coord
        self.src_distance = src_distance
        self.src_edge_type = src_edge_type
        
        self.token_pad_idx = token_pad_idx
        self.charge_pad_idx = charge_pad_idx
        
        self.split = split
        self.conf_size = conf_size
        self.left_pad = False
        self.bond_triplet = bond_triplet
        self._init_rec2mol()
        self.set_epoch(None)
        
        self.tgtnode_dataset = tgtnode_dataset
        self.tgtedge_dataset = tgtedge_dataset
        self.tgtdist_dataset = tgtdist_dataset

        self.triplet_dataset = triplet_dataset

        self.qm_dataset = qm_dataset
        
    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def _init_rec2mol(self):
        self.rec2mol = collections.defaultdict(list)
        if self.split in ['train','train.small']:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx = self.idx2key[i]
                self.rec2mol[smi_idx].append(i)
        else:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx = self.idx2key[i]
                # self.rec2mol[smi_idx].append(i)
                self.rec2mol[smi_idx].extend([i * self.conf_size + j for j in range(self.conf_size)])


    def __len__(self):
        return len(self.rec2mol)

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        mol_list = self.rec2mol[index]
        src_tokens_list = []
        src_charges_list = []
        src_coord_list = []
        src_distance_list = []
        src_edge_type_list = []
        # tgt features
        # nodes
        num_nodes_list = []
        node_features_list = []
        node_mask_list = []
        # edges
        edge_mask_list = []
        # matrix
        distance_matrix_list = []
        feature_matrix_list = []
        
        # triplet
        triplet_matrix_list = []
        qm_features_list = []

        for i in mol_list:
            src_tokens_list.append(self.src_tokens[i])
            src_charges_list.append(self.src_charges[i])
            src_coord_list.append(self.src_coord[i])
            src_distance_list.append(self.src_distance[i])
            src_edge_type_list.append(self.src_edge_type[i])
            if self.qm_dataset != None:
                qm_features_list.append(self.qm_dataset[i])
            
            num_nodes_list.append(self.tgtnode_dataset[i]['num_nodes'])
            node_features_list.append(self.tgtnode_dataset[i]['node_features'])
            node_mask_list.append(self.tgtnode_dataset[i]['node_mask'])
            edge_mask_list.append(self.tgtedge_dataset[i]['edge_mask'])
            distance_matrix_list.append(self.tgtdist_dataset[i]['distance_matrix'])
            feature_matrix_list.append(self.tgtdist_dataset[i]['feature_matrix'])


            if self.bond_triplet:
                triplet_matrix_list.append(self.triplet_dataset[i])

        return src_tokens_list, src_charges_list,src_coord_list,src_distance_list,src_edge_type_list, \
                num_nodes_list, node_features_list, node_mask_list, edge_mask_list, distance_matrix_list, feature_matrix_list, \
                triplet_matrix_list, qm_features_list
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
    
    def collater(self, samples):
        batch = [len(samples[i][0]) for i in range(len(samples))]

        src_tokens, src_charges, src_coord, src_distance, src_edge_type, \
            num_nodes, node_features, node_mask, edge_mask, distance_matrix, feature_matrix, triplet_matrix, qm_features_list = [list(chain.from_iterable(i)) for i in zip(*samples)]
        src_tokens = collate_tokens(src_tokens, self.token_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_charges = collate_tokens(src_charges, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_coord = collate_tokens_coords(src_coord, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_distance = collate_tokens_2d(src_distance, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_edge_type = collate_tokens_2d(src_edge_type, 0, left_pad=self.left_pad, pad_to_multiple=8)
        num_nodes = torch.stack(num_nodes)
        node_features = collate_tokens_2d_2(node_features, 0, left_pad=self.left_pad, pad_to_multiple=8)
        node_mask = collate_tokens(node_mask, 0, left_pad=self.left_pad, pad_to_multiple=8)
        edge_mask = collate_tokens_2d(edge_mask, 0, left_pad=self.left_pad, pad_to_multiple=8)
        distance_matrix = collate_tokens_2d(distance_matrix, 0, left_pad=self.left_pad, pad_to_multiple=8)
        feature_matrix = stack_tensor_3d(feature_matrix, pad_to_multiple=8)
        if self.qm_dataset != None:
            qm_features = torch.stack(qm_features_list)
        else:
            qm_features = None

        if self.bond_triplet: 
            triplet_matrix = stack_tensor_4d(triplet_matrix, pad_to_multiple=8)
            triplet_matrix = trimat_to_mask(triplet_matrix, num_nodes)
            # print(triplet_matrix.shape)
        else:
            triplet_matrix = None
        
        # return {'num_nodes': num_nodes, 'node_mask': node_mask, 'rdkit_coords': src_coord, 'node_features': node_features,'distance_matrix': distance_matrix, 
        #         'feature_matrix':feature_matrix, 'edge_mask': edge_mask, 'src_charges': src_charges, 'src_tokens': src_tokens, 'dist_input': src_distance,
        #         'src_tokens':src_tokens ,'src_edge_type':src_edge_type, 'token_targets': token_targets ,'charge_targets': charge_targets, 'coord_targets': coord_targets,
        #         'dist_targets':dist_targets}

        return  src_tokens, src_charges, src_coord, src_distance, src_edge_type, \
                num_nodes, node_features, node_mask, edge_mask, distance_matrix, feature_matrix, triplet_matrix, qm_features, \
                batch