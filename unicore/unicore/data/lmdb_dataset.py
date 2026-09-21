# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import os
import pickle
import torch
import numpy as np
import collections
from functools import lru_cache
from . import data_utils
import logging
logger = logging.getLogger(__name__)

class LMDBDataset:
    def __init__(self, db_path, conf_size=1):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

        self.conf_size = conf_size
        
    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        if self.conf_size > 1:
            metadata_a = data['metadata_a']
            metadata_b = data['metadata_b']
            for i, d_a in enumerate(metadata_a):
                if len(d_a['coordinates'])!= self.conf_size:
                    # print(len(d_a['coordinates']),',a:', idx, ', mol:', i)
                    data['metadata_a'][i]['coordinates'] = np.tile(data['metadata_a'][i]['coordinates'][0], (self.conf_size,1,1))
            for i, d_b in enumerate(metadata_b):
                if len(d_b['coordinates'])!= self.conf_size:
                    # print(len(d_b['coordinates']),',b:', idx,', mol:', i)
                    data['metadata_b'][i]['coordinates'] = np.tile(data['metadata_b'][i]['coordinates'][0], (self.conf_size,1,1))
        return data


