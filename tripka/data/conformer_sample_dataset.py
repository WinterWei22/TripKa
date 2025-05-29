# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils
import torch

class ConformerSamplePKADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, charges, id="ori_smi", mode = 'tgt', conf_size = 1, qm=False, with_qm=False, wiberg_thredhold=0.001):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.charges = charges
        self.id = id
        self.mode = mode
        self.conf_size = conf_size
        self.set_epoch(None)
        self.qm = qm
        self.with_qm = with_qm
        self.wiberg_thredhold = wiberg_thredhold

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        charges = np.array(self.dataset[index][self.charges])
        assert len(atoms) > 0, 'atoms: {}, charges: {}, coordinates: {}, id: {}'.format(atoms, charges, coordinates, id)
        if self.conf_size > 1:
            size = len(self.dataset[index][self.coordinates])
            with data_utils.numpy_seed(self.seed, epoch, index):
                sample_idx = np.random.randint(size)
            coordinates = self.dataset[index][self.coordinates][sample_idx]
        else:
            coordinates = self.dataset[index][self.coordinates]
        if self.mode == 'tgt':
            smi = self.dataset[index]['smi']
            if self.with_qm:
                wiberg = self.dataset[index]["wiberg"][sample_idx,:,:]
                mullikenCharges = self.dataset[index]["mullikenCharges"][sample_idx,:]
                return {"atoms": atoms, "coordinates": coordinates.astype(np.float32),"charges":charges,"id": self.id,"smi":smi, 
                        "wiberg":torch.tensor(wiberg ,dtype=torch.float16), "mullikenCharges":torch.tensor(mullikenCharges ,dtype=torch.float16),}
            else:
                return {"atoms": atoms, "coordinates": coordinates.astype(np.float32),"charges":charges,"id": self.id,"smi":smi}
        else:
            return {"atoms": atoms, "coordinates": coordinates.astype(np.float32),"charges":charges,"id": self.id}
        
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
