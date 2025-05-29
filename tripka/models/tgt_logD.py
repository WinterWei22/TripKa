# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .utils import ClassificationHead, NonLinearHead
from .tgt import TGT_Encoder
from . import layers
from . import consts as C

logger = logging.getLogger(__name__)


@register_model("tgt_logd")
class TGTLOGDModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-charge-loss",
            type=float,
            metavar="D",
            help="mask charge loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )

    def __init__(self, args,
                dictionary,
                charge_dictionary,
                model_height,
                layer_multiplier    = 1         ,
                upto_hop            = 32        ,
                embed_3d_type       = 'gaussian',
                num_3d_kernels      = 128       ,
                num_dist_bins       = 128       ,
                **layer_configs):
        super().__init__()
        tgt_pka_architecture(args)
        self.dictionary = dictionary
        self.charge_dictionary = charge_dictionary
        
        self.args = args
        self.model_height        = model_height
        self.layer_multiplier    = layer_multiplier
        self.upto_hop            = upto_hop
        self.embed_3d_type       = embed_3d_type
        self.num_3d_kernels      = num_3d_kernels
        self.num_dist_bins       = num_dist_bins
        
        self.node_width          = layer_configs['node_width']
        self.edge_width          = layer_configs['edge_width']
        
        self.layer_configs = layer_configs
        self.encoder = TGT_Encoder(model_height     = self.model_height      ,
                                   layer_multiplier = self.layer_multiplier  ,
                                   node_ended       = True                   ,
                                   edge_ended       = True                   ,
                                   egt_simple       = False                  ,
                                   **self.layer_configs)
        
        self.input_embed = layers.EmbedInput(node_width      = self.node_width     ,
                                             edge_width      = self.edge_width     ,
                                             upto_hop        = self.upto_hop       ,
                                             embed_3d_type   = self.embed_3d_type   ,
                                             num_3d_kernels  = self.num_3d_kernels  ,
                                             charge_dictionary = charge_dictionary  ,
                                            )
        
        self.final_ln_node = nn.LayerNorm(self.node_width)
        
        self.final_ln_edge = nn.LayerNorm(self.edge_width)
        
        """Heads"""
        if args.masked_token_loss > 0:
            self.token_pred = MaskLMHead(layer_configs['node_width'], len(dictionary), activation_fn='gelu')
        if args.masked_charge_loss > 0:
            self.charge_pred = MaskLMHead(layer_configs['node_width'], len(charge_dictionary), activation_fn='gelu')
        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                layer_configs['edge_width'], 1, 'gelu'
            )
        if args.masked_dist_loss > 0:
            self.dist_pred = DistanceHead(self.edge_width, 'gelu')
        
        if args.logP:
            self.logP_pred = ClassificationHead(self.node_width, self.node_width, 1,  'gelu', 0.1, mode='tgt')

        if args.logD:
            self.logd_pred = ClassificationHead(self.node_width, self.node_width, 1,  'gelu', 0.1, mode='tgt')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.charge_dictionary, **task.model_config)

    def forward(
        self,
        input_metadata,
        classification_head_name=None,
        encoder_masked_tokens=None,
        features_only=False,
        **kwargs
    ):
        if not features_only:
            src_tokens, src_charges, src_coord, src_distance, src_edge_type, \
            num_nodes, node_features, node_mask, edge_mask, distance_matrix, feature_matrix, triplet_matrix, \
            batch, charge_targets, coord_targets, dist_targets, token_targets = input_metadata
        else:
            src_tokens, src_charges, src_coord, src_distance, src_edge_type, \
            num_nodes, node_features, node_mask, edge_mask, distance_matrix, feature_matrix, triplet_matrix, qm_features, \
            batch = input_metadata
            charge_targets, coord_targets, dist_targets, token_targets = None, None, None, None
        inputs = {'num_nodes': num_nodes, 'node_mask': node_mask, 'rdkit_coords': src_coord, 'node_features': node_features,'distance_matrix': distance_matrix, 
                'feature_matrix':feature_matrix, 'edge_mask': edge_mask, 'src_charges': src_charges, 'src_tokens': src_tokens, 'dist_input': src_distance,
                'src_edge_type':src_edge_type, 'token_targets': token_targets ,'charge_targets': charge_targets, 'coord_targets': coord_targets,
                'dist_targets':dist_targets, 'triplet_matrix': triplet_matrix}
        
        g = self.input_embed(inputs)
        g = self.encoder(g)
        h = g.h
        h = self.final_ln_node(h)
        
        nodem = g.node_mask.half().unsqueeze(dim=-1)
        h = (h*nodem).sum(dim=1)/(nodem.sum(dim=1)+1e-9)

        
        if self.args.logD:
            logD_pred = self.logd_pred(h).squeeze(dim=-1)
            if self.args.logP:
                logP_pred = self.logP_pred(h).squeeze(dim=-1)
            else:
                logP_pred = torch.zeros(logD_pred.unsqueeze(1).shape, device=logD_pred.device).squeeze(dim=-1)
        else:
            logP_pred = self.logP_pred(h).squeeze(dim=-1)
            logD_pred = torch.zeros(logP_pred.unsqueeze(1).shape, device=logP_pred.device).squeeze(dim=-1)

        cls_logits = torch.cat((logD_pred.unsqueeze(1), logP_pred.unsqueeze(1)), dim=1)

        return cls_logits, batch

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@register_model_architecture("tgt_logd", "tgt_logd")
def tgt_pka_architecture(args):
    def base_architecture(args):
        args.model_height = getattr(args, "model_height", 15)
        args.layer_multiplier = getattr(args, "layer_multiplier", 1)
        args.upto_hop = getattr(args, "upto_hop", 32)
        args.embed_3d_type = getattr(args, "embed_3d_type", 'gaussian')
        args.num_3d_kernels = getattr(args, "num_3d_kernels", 128)
        args.num_dist_bins = getattr(args, "num_dist_bins", 128)
        args.encoder_layers = getattr(args, "encoder_layers", 15)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
        args.dropout = getattr(args, "dropout", 0.1)
        args.emb_dropout = getattr(args, "emb_dropout", 0.1)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.activation_dropout = getattr(args, "activation_dropout", 0.0)
        args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
        args.max_seq_len = getattr(args, "max_seq_len", 512)
        args.activation_fn = getattr(args, "activation_fn", "gelu")
        args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
        args.post_ln = getattr(args, "post_ln", False)
        args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
        args.masked_charge_loss = getattr(args, "masked_charge_loss", -1.0)
        args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
        args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)    
        args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
        args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
        
    base_architecture(args)
