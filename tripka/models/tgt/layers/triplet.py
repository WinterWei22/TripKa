import torch
from torch import nn
import torch.nn.functional as F
from .utils import *

def get_triplet_layer(layer_type):
    if layer_type == 'tiangular_update':
        return TriangularUpdate
    elif layer_type == 'linear':
        return linear_layer

    else:
        raise ValueError(f'Invalid layer_type: {layer_type}')
        
@torch.jit.script
def siglin(gates, lins):
    return torch.sigmoid(gates) * lins

class TriangularUpdate(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 attention_dropout = 0 ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.attention_dropout   = attention_dropout
             
        self.tri_ln_e   = nn.LayerNorm(self.edge_width)
        
        self.lin_V   = nn.Linear(self.edge_width, self.num_heads*4)
        self.lin_E  = nn.Linear(self.edge_width, self.num_heads*4)
        
        self.lin_O  = nn.Linear(self.num_heads*2, self.edge_width*2)
    
    def forward(self, e, mask, triplet_matrix=None):
        e_ln = self.tri_ln_e(e)
        
        # Projections
        V_in_g, V_in_l, V_out_g, V_out_l = self.lin_V(e_ln).chunk(4, dim=-1)
        E_in_g, E_in_l, E_out_g, E_out_l = self.lin_E(e_ln).chunk(4, dim=-1)
        
        E_in_g = E_in_g + mask
        E_out_g = E_out_g + mask
        V_in_g = V_in_g + mask
        V_out_g = V_out_g + mask
        
        V_in = siglin(V_in_g, V_in_l)
        V_out = siglin(V_out_g, V_out_l)
        E_in = siglin(E_in_g, E_in_l)
        E_out = siglin(E_out_g, E_out_l)
        
        Va_in = torch.einsum('bikh,bjkh->bijh', E_in, V_in)
        Va_out = torch.einsum('bkih,bkjh->bijh', E_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1)
        
        e_g, e_l = self.lin_O(Va).chunk(2, dim=-1)
        e = siglin(e_g, e_l)
        return e

       
class linear_layer(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 attention_dropout = 0 ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.attention_dropout   = attention_dropout
        
        assert not (self.edge_width % self.num_heads),\
                'edge_width must be divisible by num_heads'    
        self.tri_ln_e   = nn.LayerNorm(self.edge_width)
        
        self.lin_O  = nn.Linear(self.edge_width, self.edge_width)
    
    def forward(self, e, mask, triplet_matrix=None):
        e_ln = self.tri_ln_e(e)
        e = self.lin_O(e_ln)

        return e