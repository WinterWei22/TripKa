# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import torch

class TTADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, charges, id="ori_smi", mode='tgt', conf_size=10, qm=False, with_qm=False, wiberg_thredhold=0.001):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.charges = charges
        self.id = id
        self.conf_size = conf_size
        self.mode = mode
        self.qm = qm
        self.with_qm = with_qm
        self.wiberg_thredhold = wiberg_thredhold
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        mol_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[mol_idx][self.atoms])
        charges = np.array(self.dataset[mol_idx][self.charges])
        if self.conf_size > 1:
            coordinates = np.array(self.dataset[mol_idx][self.coordinates][coord_idx])
        else:
            # if self.mode != 'tgt':
            #     coordinates = np.array(self.dataset[mol_idx][self.coordinates][coord_idx])  # !
            # else:
            coordinates = np.array(self.dataset[mol_idx][self.coordinates])
        id = self.dataset[mol_idx][self.id]
        target = self.dataset[mol_idx]["target"]
        if self.mode == 'tgt':
            smi = self.dataset[mol_idx]['smi']
        else:
            smi = self.dataset[mol_idx][self.id]
        if self.with_qm:
            wiberg = self.dataset[mol_idx]["wiberg"][coord_idx,:,:]
            mullikenCharges = self.dataset[mol_idx]["mullikenCharges"][coord_idx,:]
            return {
                "atoms": atoms,
                "coordinates": coordinates.astype(np.float32),
                "charges": charges.astype(str),
                "target": target,
                "smi":smi,
                "wiberg":torch.tensor(wiberg ,dtype=torch.float16),
                "mullikenCharges":torch.tensor(mullikenCharges ,dtype=torch.float16),
                "target": target,
                "id": id,
            }
        else:
            return {
                "atoms": atoms,
                "coordinates": coordinates.astype(np.float32),
                "charges": charges.astype(str),
                "target": target,
                "smi":smi,
                "target": target,
                "id": id,
            }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class TTAPKADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, metadata, atoms, coordinates, charges, id="ori_smi", mode='tgt', qm=False, with_qm=False):
        self.dataset = dataset
        self.seed = seed
        self.metadata = metadata
        self.atoms = atoms
        self.coordinates = coordinates
        self.charges = charges
        self.id = id
        self.mode = mode
        self.qm = qm
        self.with_qm = with_qm

        self._init_idx()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def _init_idx(self):
        self.idx2key = {}
        total_sz = 0
        for i in range(len(self.dataset)):
            size = len(self.dataset[i][self.metadata])
            for j in range(size):
                self.idx2key[total_sz] = (i, j)
                total_sz += 1
        self.total_sz = total_sz

    def get_idx2key(self):
        return self.idx2key

    def __len__(self):
        return self.total_sz

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx, mol_idx = self.idx2key[index]
        atoms = np.array(self.dataset[smi_idx][self.metadata][mol_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.metadata][mol_idx][self.coordinates])
        charges = np.array(self.dataset[smi_idx][self.metadata][mol_idx][self.charges])
        if self.mode == 'tgt':
            smi = self.dataset[smi_idx][self.metadata][mol_idx]["smi"]
        else:
            smi = self.dataset[smi_idx]["ori_smi"]
        
        id = self.dataset[smi_idx][self.id]
        target = self.dataset[smi_idx]["target"]
        if self.with_qm:
            wiberg = self.dataset[smi_idx][self.metadata][mol_idx]['wiberg']
            mullikenCharges = self.dataset[smi_idx][self.metadata][mol_idx]['mullikenCharges']
            return {
                "atoms": atoms,
                "coordinates": coordinates.astype(np.float32),
                "charges": charges.astype(str),
                "smi": smi,
                "target": target,
                "wiberg": wiberg,
                "mullikenCharges":mullikenCharges,
                "id": id,
            }
        else:
            return {
                "atoms": atoms,
                "coordinates": coordinates.astype(np.float32),
                "charges": charges.astype(str),
                "smi": smi,
                "target": target,
                "id": id,
            }
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class TTALOGDDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, metadata, atoms, coordinates, charges, id="ori_smi", mode='tgt', qm=False):
        self.dataset = dataset
        self.seed = seed
        self.metadata = metadata
        self.atoms = atoms
        self.coordinates = coordinates
        self.charges = charges
        self.id = id
        self.mode = mode
        self.qm = qm

        self._init_idx()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def _init_idx(self):
        self.idx2key = {}
        total_sz = 0
        for i in range(len(self.dataset)):
            self.idx2key[total_sz] = i
            total_sz += 1
        self.total_sz = total_sz

    def get_idx2key(self):
        return self.idx2key

    def __len__(self):
        return self.total_sz

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = self.idx2key[index]
        atoms = np.array(self.dataset[smi_idx][self.metadata][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.metadata][self.coordinates])
        charges = np.array(self.dataset[smi_idx][self.metadata][self.charges])
        if self.mode == 'tgt':
            smi = self.dataset[smi_idx][self.metadata]["smi"]
        else:
            smi = self.dataset[smi_idx]["ori_smi"]
        
        id = self.dataset[smi_idx][self.id]
        target = self.dataset[smi_idx]["target"]
        if self.qm:
            qm_features = self.dataset[smi_idx]["qm_features"]
            return {
                "atoms": atoms,
                "coordinates": coordinates.astype(np.float32),
                "charges": charges.astype(str),
                "smi": smi,
                "target": target,
                "qm_features": qm_features,
                "id": id,
            }
        else:
            return {
                "atoms": atoms,
                "coordinates": coordinates.astype(np.float32),
                "charges": charges.astype(str),
                "smi": smi,
                "target": target,
                "id": id,
            }
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)