# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    SortDataset,
    TokenizeDataset,
    RawLabelDataset,
    FromNumpyDataset,
)
from tripka.data import (
    KeyDataset,
    ConformerSamplePKADataset,
    PKAInputDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    NormalizeDataset,
    CroppingDataset,
    FoldLMDBDataset,
    StackedLMDBDataset,
    SplitLMDBDataset,
    data_utils,
    DistMatDataset,
    TGTEdgeDataset,
    TGTNodeDataset,
    TGT_PKAMLMInputDataset,
    TGT_PKAInputDataset,
    TGTtriDataset
)

from tripka.data.tta_dataset import TTADataset, TTAPKADataset
from unicore.tasks import UnicoreTask, register_task
import random
import yaml
from yaml import SafeLoader as yaml_Loader, SafeDumper as yaml_Dumper
logger = logging.getLogger(__name__)


@register_task("tgt_pka")
class TGTMolPKATask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--charge-dict-name",
            default="dict_charge.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )
        parser.add_argument(
            '--split-mode', 
            type=str, 
            default='predefine',
            choices=['predefine', 'cross_valid', 'random', 'infer'],
        )
        parser.add_argument(
            "--nfolds",
            default=5,
            type=int,
            help="cross validation split folds"
        )
        parser.add_argument(
            "--fold",
            default=0,
            type=int,
            help='local fold used as validation set, and other folds will be used as train set'
        )
        parser.add_argument(
            "--cv-seed",
            default=42,
            type=int,
            help="random seed used to do cross validation splits"
        )
        parser.add_argument(
            "--tgt-config",
            default='config/tgt_test.yaml',
            type=str,
            help="tgt model config"
        )
        parser.add_argument(
            "--accumulate",
            default=1,
            type=int,
            help="accumulation steps"
        )
        
        parser.add_argument(
            "--bond-triplet",
            default=False,
            action="store_true",
            help="bond triplet"
        )

        parser.add_argument(
            "--qm",
            default=False,
            action="store_true",
            help="qm features"
        )

        parser.add_argument(
            "--with_charges",
            default=False,
            action="store_true",
            help="with charges"
        )

        parser.add_argument(
            "--bond-type",
            default='hole',
            type=str,
            help="bond type: hole, bond or radius"
        )

        parser.add_argument(
            "--confidence",
            default=False,
            action="store_true",
            help="use confidence model"
        )

        parser.add_argument(
            "--get-emb",
            default=False,
            action="store_true",
            help="get molecular embeddings"
        )

        parser.add_argument(
            "--get-edge-weights",
            default=False,
            action="store_true",
            help="get edges weights"
        )

        parser.add_argument(
            "--get-attn-weights",
            default=False,
            action="store_true",
            help="get attention weights"
        )

        parser.add_argument(
            "--confidence-train",
            default=False,
            action="store_true",
            help="train the confidence model"
        )

        parser.add_argument(
            "--thredhold",
            default=0.15,
            type=float,
            help="confidence threadhold"
        )

        parser.add_argument(
            "--with-qm",
            default=False,
            action="store_true",
            help="finetune with qm features"      
        )

        parser.add_argument(
            "--load-mean",
            default=False,
            action="store_true",
            help="load origin mean while finetuning again"      
        )

        parser.add_argument(
            "--wiberg-thredhold",
            default=0.001,
            type=float,
            help="wiberg thredhold"
        )

    def __init__(self, args, dictionary, charge_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.charge_dictionary = charge_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.charge_mask_idx = charge_dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
        if self.args.split_mode !='predefine':
            self.__init_data()
        
        # load tgt model config    
        self.model_config = self.get_tgt_config(args)

        # set seed manually
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)    
        print(args.seed)
    def __init_data(self):
        data_path = os.path.join(self.args.data, self.args.task_name, self.args.task_name + '.lmdb')
        raw_dataset = LMDBDataset(data_path)
        if self.args.split_mode == 'cross_valid':
            train_folds = []
            for _fold in range(self.args.nfolds):
                if _fold == 0:
                    cache_fold_info = FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds).get_fold_info()
                if _fold == self.args.fold:
                    self.valid_dataset = FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info)
                if _fold != self.args.fold:
                    train_folds.append(FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info))
            self.train_dataset = StackedLMDBDataset(train_folds)
        elif self.args.split_mode == 'random':
            cache_fold_info = SplitLMDBDataset(raw_dataset, self.args.seed, 0).get_fold_info()   
            self.train_dataset = SplitLMDBDataset(raw_dataset, self.args.seed, 0, cache_fold_info=cache_fold_info)
            self.valid_dataset = SplitLMDBDataset(raw_dataset, self.args.seed, 1, cache_fold_info=cache_fold_info)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        charge_dictionary = Dictionary.load(os.path.join(args.data, args.charge_dict_name))
        logger.info("charge dictionary: {} types".format(len(charge_dictionary)))
        return cls(args, dictionary, charge_dictionary)
    
    def get_tgt_config(self, args):
        # load config yaml

        
        reload_config_path = ''
        if hasattr(args, 'save_dir'):
            save_dir = args.save_dir
            reload_config_path = f'{save_dir}/model.yaml'
        load_path = args.tgt_config if not os.path.exists(reload_config_path) else reload_config_path
        
        with open(load_path, 'r') as fp:
             config = yaml.load(fp, Loader=yaml_Loader)
        model_config = dict(
            model_height        = config["model_height"]                ,
            node_width          = config["node_width"]                  ,
            edge_width          = config["edge_width"]                   ,
            num_heads           = config["num_heads"]                    ,
            node_act_dropout    = config["node_act_dropout"]             ,
            edge_act_dropout    = config["edge_act_dropout"]             ,
            source_dropout      = config["source_dropout"]               ,
            drop_path           = config["drop_path"]                    ,
            activation          = config["activation"]                   ,
            scale_degree        = config["scale_degree"]                 ,
            node_ffn_multiplier = config["node_ffn_multiplier"]          ,
            edge_ffn_multiplier = config["edge_ffn_multiplier"]          ,
            layer_multiplier    = 1,    # config["layer_multiplier"]             ,
            upto_hop            = config["upto_hop"]                     ,
            triplet_heads       = config["triplet_heads"]                ,
            triplet_type        = config["triplet_type"]                 ,
            triplet_dropout     = 0,     # config["triplet_dropout"]              ,
            
            num_3d_kernels      = 128,  # config["num_3d_kernels"]               ,
            embed_3d_type       = 'gaussian',   # config["embed_3d_type"]                ,
        )
        
        if hasattr(args, 'save_dir'):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            with open(reload_config_path, 'w', encoding='utf-8') as file:
                yaml.dump(model_config, file, allow_unicode=True, default_flow_style=False)
                
        return model_config
    
    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        self.split = split
        if self.args.split_mode != 'predefine':
            if split == 'train':
                dataset = self.train_dataset
            elif split == 'valid':
                dataset = self.valid_dataset
        else:
            split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
            dataset = LMDBDataset(split_path, self.args.conf_size)
        tgt_dataset = KeyDataset(dataset, "target")
        if split in ['train', 'train.small']:
            tgt_list = [tgt_dataset[i] for i in range(len(tgt_dataset))]
            if self.args.load_mean:
                self.mean = self.load_mean()
            else:
                self.mean = sum(tgt_list) / len(tgt_list)
            self.write_mean()
            self.std = 1
        # elif split in ['sampl8_macro_regen_2', 'sampl6_micro', 'dwar_small', 'novartis_a', 'novartis_b', 'sampl7_qm', 'EpKa','sampl6_macro_regen', 'sampl7_macro_regen', 'sampl8_macro_regen','az', 'sampl7_macro_regen_qm','sampl7_macro_regen_conf1_MM','sampl7_macro_regen_conf6_MM']:
        else:
            # self.mean = 6.504894871171601  # precompute from dwar_8228 full set !!!!!!部分不同conf dwar数据量不同
            self.mean = self.load_mean()
            self.std = 1
        id_dataset = KeyDataset(dataset, "ori_smi")

        def GetPKAInput(dataset, metadata_key):

            mol_dataset = TTAPKADataset(dataset, self.args.seed, metadata_key, "atoms", "coordinates", "charges", mode='tgt', qm=self.args.qm, with_qm = self.args.with_qm)
            idx2key = mol_dataset.get_idx2key()
            if split in ["train","train.small"]:
                sample_dataset = ConformerSamplePKADataset(
                    mol_dataset, self.args.seed, "atoms", "coordinates", "charges", mode='tgt', conf_size=self.args.conf_size, qm=self.args.qm, 
                    with_qm = self.args.with_qm, wiberg_thredhold=self.args.wiberg_thredhold
                )
            else:
                sample_dataset = TTADataset(
                    mol_dataset, self.args.seed, "atoms", "coordinates","charges","id", conf_size=self.args.conf_size, mode='tgt', qm=self.args.qm, 
                    with_qm = self.args.with_qm, wiberg_thredhold=self.args.wiberg_thredhold
                )

            sample_dataset = RemoveHydrogenDataset(
                sample_dataset,
                "atoms",
                "coordinates",
                "charges",
                self.args.remove_hydrogen,
                self.args.remove_polar_hydrogen,
            )
            sample_dataset = CroppingDataset(
                sample_dataset, self.seed, "atoms", "coordinates","charges", self.args.max_atoms
            )
            sample_dataset = NormalizeDataset(sample_dataset, "coordinates", normalize_coord=True)
            src_dataset = KeyDataset(sample_dataset, "atoms")
            smi_dataset = KeyDataset(sample_dataset, "smi")
                
            src_dataset = TokenizeDataset(
                src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
            )
            src_charge_dataset = KeyDataset(sample_dataset, "charges")
            src_charge_dataset = TokenizeDataset(
                src_charge_dataset, self.charge_dictionary, max_seq_len=self.args.max_seq_len
            )
            coord_dataset = KeyDataset(sample_dataset, "coordinates")

            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            coord_dataset = FromNumpyDataset(coord_dataset)
            # coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            distance_dataset = DistanceDataset(coord_dataset)

            # tgt features
            tgtnode_dataset = TGTNodeDataset(smi_dataset)
            tgtedge_dataset = TGTEdgeDataset(tgtnode_dataset, coord_dataset=coord_dataset, bond_type=self.args.bond_type)
            distmat_dataset = DistMatDataset(tgtnode_dataset, tgtedge_dataset, coord_dataset)
            
            triplet_dataset = None
            # bond_triplet
            if self.args.bond_triplet:
                triplet_dataset = TGTtriDataset(edge_dataset=tgtedge_dataset, coord_dataset=coord_dataset, node_dataset=tgtnode_dataset)
            qm_dataset = None
            if self.args.qm:
                qm_dataset = KeyDataset(sample_dataset, "qm_features")

            wiberg_bonds = None
            mulliken_charges = None
            if self.args.with_qm:
                wiberg_bonds = KeyDataset(sample_dataset, "wiberg")
                mulliken_charges = KeyDataset(sample_dataset, "mullikenCharges")
            
            return TGT_PKAInputDataset(idx2key, src_dataset, src_charge_dataset, coord_dataset, distance_dataset, edge_type, self.dictionary.pad(), self.charge_dictionary.pad(),
                                       tgtnode_dataset,tgtedge_dataset, distmat_dataset, triplet_dataset, split, self.args.conf_size, self.args.bond_triplet, qm_dataset=qm_dataset,
                                       wiberg_bonds=wiberg_bonds, mulliken_charges=mulliken_charges,  wiberg_thredhold=self.args.wiberg_thredhold)

        input_a_dataset = GetPKAInput(dataset, "metadata_a")
        input_b_dataset = GetPKAInput(dataset, "metadata_b")

        nest_dataset = NestedDictionaryDataset(
                {
                    "net_input_a": input_a_dataset,
                    "net_input_b": input_b_dataset,
                    "target": {
                        "finetune_target": RawLabelDataset(tgt_dataset),
                    },
                    "id": id_dataset,
                },
            )

        if not self.args.no_shuffle and split in ["train","train.small"]:
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(id_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def write_mean(self):
        save_dir = self.args.save_dir
        mean_file = f'{save_dir}/model.yaml'
        with open(mean_file, "r") as file:
            data = yaml.safe_load(file)

        data["mean"] = self.mean

        with open(mean_file, "w") as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

    def load_mean(self):
        mean_file = self.args.tgt_config

        with open(mean_file, "r") as file:
            data = yaml.safe_load(file)

        mean = data['mean']

        return mean

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)

        if args.confidence and self.args.confidence_train:
            for name, param in model.named_parameters():
                param.requires_grad = False

            model.confidence_pred = model.create_MLP()
        
        return model
    
    # def train_step(
    #     self, sample, model, loss, optimizer, update_num, ignore_grad=False
    # ):
    #     """
    #     Do forward and backward, and return the loss as computed by *loss*
    #     for the given *model* and *sample*.

    #     Args:
    #         sample (dict): the mini-batch. The format is defined by the
    #             :class:`~unicore.data.UnicoreDataset`.
    #         model (~unicore.models.BaseUnicoreModel): the model
    #         loss (~unicore.losses.UnicoreLoss): the loss
    #         optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
    #         update_num (int): the current update
    #         ignore_grad (bool): multiply loss by 0 if this is set to True

    #     Returns:
    #         tuple:
    #             - the loss
    #             - the sample size, which is used as the denominator for the
    #               gradient
    #             - logging outputs to display while training
    #     """
    #     model.train()
    #     model.set_num_updates(update_num)
    #     with torch.autograd.profiler.record_function("forward"):
    #         loss, sample_size, logging_output = loss(model, sample)
    #         loss = loss / self.args.accumulate
    #     if ignore_grad:
    #         loss *= 0
        
    #     with torch.autograd.profiler.record_function("backward"):
    #         optimizer.backward(loss)
                  
    #     return loss, sample_size, logging_output
    
    # def valid_step(self, sample, model, loss, test=False):
    #     model.eval()
    #     with torch.no_grad():
    #         loss, sample_size, logging_output = loss(model, sample)
    #     return loss, sample_size, logging_output