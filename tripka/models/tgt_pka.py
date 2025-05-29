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


@register_model("tgt_pka")
class TGTPKAModel(BaseUnicoreModel):
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
        parser.add_argument(
            "--mulliken-charge-loss",
            type=float,
            metavar="D",
            help="mulliken charges loss ratio",
        )
        parser.add_argument(
            "--wiberg-bonds-loss",
            type=float,
            metavar="D",
            help="wiberg bonds loss ratio",
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
        # self.pred = nn.Linear(self.node_width, 1)
        mode = 'confidence' if self.args.confidence else 'tgt'
        mode = 'get_emb' if self.args.get_emb else mode
        self.mode = mode

        self.pred = ClassificationHead(self.node_width, self.node_width, 1,  'gelu', 0.1, mode=mode)
        # nn.init.constant_(self.pred.bias, C.HL_MEAN)
        
        self.final_ln_edge = nn.LayerNorm(self.edge_width)
        
        """Heads"""
        if args.masked_token_loss > 0:
            self.token_pred = MaskLMHead(layer_configs['node_width'], len(dictionary), activation_fn='gelu')  # 需要注意token是否加入了[CLS]等符号
        if args.masked_charge_loss > 0:
            self.charge_pred = MaskLMHead(layer_configs['node_width'], len(charge_dictionary), activation_fn='gelu')  # 需要注意charge是否加入了[CLS]等符号
        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                layer_configs['edge_width'], 1, 'gelu'
            )
        if args.masked_dist_loss > 0:
            self.dist_pred = DistanceHead(self.edge_width, 'gelu')
        
        if args.with_qm:
            self.charge_pred = MaskLMHead(layer_configs['node_width'], len(charge_dictionary), activation_fn='gelu')
            self.dist_pred = WibergHead(self.edge_width, 'gelu')
            self.create_mulliken_pred(charge_dictionary)

        if args.confidence and not args.confidence_train:
            self.confidence_pred = self.create_MLP()

        # if args.with_charges:
        #     self.charge_embed = nn.Embedding(len(charge_dictionary),
        #                                 self.node_width, padding_idx=0)
        #     self.charge_pka_pred = ClassificationHead(self.node_width, self.node_width, 1,  'gelu', 0.2, mode='tgt')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.charge_dictionary, **task.model_config)

    def create_mulliken_pred(self,charge_dictionary):
        self.mulliken_dense = nn.Linear(len(charge_dictionary), len(charge_dictionary))
        self.activation_fn = utils.get_activation_fn('gelu')
        self.dropout = nn.Dropout(p=0.3)
        self.mulliken_pred = nn.Linear(len(charge_dictionary), 1)
    
    def mulliken_mlp(self, h):
        h = self.charge_pred(h)
        h = self.mulliken_dense(h)
        h = self.activation_fn(h)
        h = self.dropout(h)
        mulliken_pred = self.mulliken_pred(h).squeeze(-1)
        return mulliken_pred

    def create_MLP(self,):
        return ClassificationHead(self.node_width, self.node_width, 1,  'gelu', 0.261357, mode='confidence pred')

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
            wiberg_bonds, mulliken_charges, batch = input_metadata
            charge_targets, coord_targets, dist_targets, token_targets = None, None, None, None
        inputs = {'num_nodes': num_nodes, 'node_mask': node_mask, 'rdkit_coords': src_coord, 'node_features': node_features,'distance_matrix': distance_matrix, 
                'feature_matrix':feature_matrix, 'edge_mask': edge_mask, 'src_charges': src_charges, 'src_tokens': src_tokens, 'dist_input': src_distance,
                'src_edge_type':src_edge_type, 'token_targets': token_targets ,'charge_targets': charge_targets, 'coord_targets': coord_targets,
                'dist_targets':dist_targets, 'triplet_matrix': triplet_matrix}
        encoder_coord = None
        logits = None
        charge_logits = None
        encoder_distance = None
        x_norm = None
        delta_encoder_pair_rep_norm = None
        inputs['w'] = None

        g = self.input_embed(inputs)
        g = self.encoder(g)
        h = g.h
        h = self.final_ln_node(h)

        if self.args.get_edge_weights:
            return g.e, batch

        if self.args.get_attn_weights:
            return g.w, batch

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.token_pred(h, encoder_masked_tokens)
            if self.args.masked_charge_loss > 0:
                charge_logits = self.charge_pred(h, encoder_masked_tokens)
            if self.args.masked_coord_loss > 0 or self.args.masked_dist_loss > 0:
                e = g.e
                e = self.final_ln_edge(e)
                if self.args.masked_coord_loss > 0:
                    coords_emb = src_coord
                    delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                    attn_probs = self.pair2coord_proj(e)
                    coord_update = delta_pos / num_nodes.view(-1,1,1,1) * attn_probs
                    coord_update = torch.sum(coord_update, dim=2)
                    encoder_coord = coords_emb + coord_update
                if self.args.masked_dist_loss > 0:
                    encoder_distance = self.dist_pred(e)
        
        if self.args.with_qm:
            mulliken_pred = self.mulliken_mlp(h)
            e = g.e
            e = self.final_ln_edge(e)
            wiberg_pred = self.dist_pred(e)

        nodem = g.node_mask.half().unsqueeze(dim=-1)
        h = (h*nodem).sum(dim=1)/(nodem.sum(dim=1)+1e-9)
        if self.args.confidence:
            cls_logits, c_h = self.pred(h)
            cls_logits = cls_logits.squeeze(dim=-1)
        else:
            cls_logits = self.pred(h).squeeze(dim=-1)
            
        if self.args.qm:
            h = torch.cat([h, qm_features], dim=-1)
            cls_logits = (cls_logits + self.qm_pred(h).squeeze(dim=-1)) / 2

        if self.args.confidence:
            # confidence_pred = F.softplus(self.confidence_pred(h)).squeeze(dim=-1) 
            confidence_pred = self.confidence_pred(c_h).squeeze(dim=-1) 
            cls_logits = torch.cat((cls_logits.unsqueeze(1), confidence_pred.unsqueeze(1)), dim=1)
            
        if not features_only:
            return (
                cls_logits, batch,
                logits, charge_logits, encoder_distance, encoder_coord, x_norm, delta_encoder_pair_rep_norm,
                token_targets, charge_targets, coord_targets, dist_targets
            )
        else:
            if self.args.with_qm:
                return (cls_logits, batch, mulliken_pred, wiberg_pred, mulliken_charges, wiberg_bonds)
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
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
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
    
class WibergHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 3)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len, 3)
        # x = (x + x.transpose(-1, -2)) * 0.5
        return x

@register_model_architecture("tgt_pka", "tgt_pka")
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
