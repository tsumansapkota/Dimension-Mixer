import torch
import torch.nn as nn
import numpy as np


#############################################################################
#############################################################################

class ResMlpBlock(nn.Module):
    
    def __init__(self, input_dim, hidden_layers_ratio=[2], actf=nn.GELU):
        super().__init__()
        self.input_dim = input_dim
        #### convert hidden layers ratio to list if integer is inputted
        if isinstance(hidden_layers_ratio, int):
            hidden_layers_ratio = [hidden_layers_ratio]
            
        self.hlr = [1]+hidden_layers_ratio+[1]
        
        self.mlp = []
        ### for 1 hidden layer, we iterate 2 times
        for h in range(len(self.hlr)-1):
            i, o = int(self.hlr[h]*self.input_dim),\
                    int(self.hlr[h+1]*self.input_dim)
            self.mlp.append(nn.Linear(i, o))
            self.mlp.append(actf())
        self.mlp = self.mlp[:-1]
        
        self.mlp = nn.Sequential(*self.mlp)
        
    def forward(self, x):
        return self.mlp(x)+x
    
    
#############################################################################
#############################################################################
    
    
# class SelfAttention(nn.Module):
#     def __init__(self, embed_size, heads):
#         super(SelfAttention, self).__init__()
#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads
#
#         assert (
#             self.head_dim * heads == embed_size
#         ), "Embedding size needs to be divisible by heads"
#
#         self.values = nn.Linear(embed_size, embed_size)
#         self.keys = nn.Linear(embed_size, embed_size)
#         self.queries = nn.Linear(embed_size, embed_size)
#         self.fc_out = nn.Linear(embed_size, embed_size)
#
#
#
#     def forward(self, values, keys, query, mask):
#         # Get number of training examples
#         N = query.shape[0]
#
#         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
#
#         values = self.values(values)  # (N, value_len, embed_size)
#         keys = self.keys(keys)  # (N, key_len, embed_size)
#         queries = self.queries(query)  # (N, query_len, embed_size)
#
#         # Split the embedding into self.heads different pieces
#         values = values.reshape(N, value_len, self.heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
#         queries = queries.reshape(N, query_len, self.heads, self.head_dim)
#
#         energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
#         del queries, keys
# #         attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
#
#         # queries shape: (N, query_len, heads, heads_dim),
#         # keys shape: (N, key_len, heads, heads_dim)
#         # energy: (N, heads, query_len, key_len)
#
#         # Mask padded indices so their weights become 0
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))
#
#         attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
#         del energy
#
#         # attention shape: (N, heads, query_len, key_len)
#
#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
#             N, query_len, self.heads * self.head_dim
#         )
#
#         del attention, values
#         # attention shape: (N, heads, query_len, key_len)
#         # values shape: (N, value_len, heads, heads_dim)
#         # out after matrix multiply: (N, query_len, heads, head_dim), then
#         # we reshape and flatten the last two dimensions.
#
#         out = self.fc_out(out)
#         # Linear layer doesn't modify the shape, final shape will be
#         # (N, query_len, embed_size)
#
#         return out
    
    
#############################################################################
#############################################################################


# class TransformerBlock(nn.Module):
#     def __init__(self, embed_size, heads, dropout, forward_expansion, actf=nn.GELU):
#         super(TransformerBlock, self).__init__()
#
#         self.attention = SelfAttention(embed_size, heads)
#         self.norm1 = nn.LayerNorm(embed_size)
#         self.norm2 = nn.LayerNorm(embed_size)
#
#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_size, int(forward_expansion * embed_size)),
#             actf(),
#             nn.Linear(int(forward_expansion * embed_size), embed_size),
#         )
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, query):
#         attention = self.attention(query, query, query, None)
#
#         # Add skip connection, run through normalization and finally dropout
#         x = self.dropout(self.norm1(attention + query))
#         forward = self.feed_forward(x)
#         out = self.dropout(self.norm2(forward + x))
#         return out


#############################################################################
#############################################################################


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

    
#############################################################################
#############################################################################
############### CUSTOM FOR SPARSE TRANSFORMER #########################

# class ViT_Classifier(nn.Module):
#
#     def __init__(self, image_dim:tuple, patch_size:tuple, hidden_expansion:float, num_blocks:int, num_classes:int, pos_emb = True):
#         super().__init__()
#
#         self.img_dim = image_dim ### must contain (C, H, W) or (H, W)
#
#         ### find patch dim
#         d0 = int(image_dim[-2]/patch_size[0])
#         d1 = int(image_dim[-1]/patch_size[1])
#         assert d0*patch_size[0]==image_dim[-2], "Image must be divisible into patch size"
#         assert d1*patch_size[1]==image_dim[-1], "Image must be divisible into patch size"
# #         self.d0, self.d1 = d0, d1 ### number of patches in each axis
#         __patch_size = patch_size[0]*patch_size[1]*image_dim[0] ## number of channels in each patch
#
#         ### find channel dim
#         channel_size = d0*d1 ## number of patches
#
#         ### after the number of channels are changed
#         init_dim = __patch_size
#         final_dim = int(__patch_size*hidden_expansion/2)*2
#         self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
#         #### rescale the patches (patch wise image non preserving transform, unlike bilinear interpolation)
#         self.channel_change = nn.Linear(init_dim, final_dim)
#         print(f"ViT Mixer : Channes per patch -> Initial:{init_dim} Final:{final_dim}")
#
#
#         self.channel_dim = final_dim
#         self.patch_dim = channel_size
#
#         self.transformer_blocks = []
#
#         f = self.get_factors(self.channel_dim)
#         print(f)
#         ### get number of heads close to the square root of the channel dim
#         ### n_model = n_heads*heads_dim (= channel dim)
#         fi = np.abs(np.array(f) - np.sqrt(self.channel_dim)).argmin()
#
#         _n_heads = f[fi]
#
#         print(self.channel_dim, _n_heads)
#         for i in range(num_blocks):
#             L = TransformerBlock(self.channel_dim, _n_heads, 0, 2)
#             self.transformer_blocks.append(L)
#         self.transformer_blocks = nn.Sequential(*self.transformer_blocks)
#
#         self.linear = nn.Linear(self.patch_dim*self.channel_dim, num_classes)
#
#         self.positional_encoding = nn.Identity()
#         if pos_emb:
#             self.positional_encoding = PositionalEncoding(self.channel_dim, dropout=0)
#
#     def get_factors(self, n):
#         facts = []
#         for i in range(2, n+1):
#             if n%i == 0:
#                 facts.append(i)
#         return facts
#
#     def forward(self, x):
#         bs = x.shape[0]
#         x = self.unfold(x).swapaxes(-1, -2)
#         x = self.channel_change(x)
#         x = self.positional_encoding(x)
#         x = self.transformer_blocks(x)
#         x = self.linear(x.view(bs, -1))
#         return x
#



#############################################################################
#############################################################################

class SelfAttention_Sparse(nn.Module):
    def __init__(self, embed_size, heads, use_wout=True):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.use_wout = use_wout

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

        if self.use_wout:
            self.fc_out = nn.Linear(embed_size, embed_size)

    def __repr__(self):
        S = f'SelfAttention Sparse: [embed:{self.embed_size} heads:{self.heads}]'
        return S
    
    def forward(self, values, keys, query, mask, block_size):
        # Get number of training examples
        N = query.shape[0]
        
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into 'heads' different pieces and split sequence into 'num blocks'
        values = values.reshape(N, value_len//block_size, block_size, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len//block_size, block_size, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len//block_size, block_size, self.heads, self.head_dim)

        energy = torch.einsum("n aq h d , n ak h d -> n h a qk", [queries, keys])
        # queries shape: (N, n_query_blocks, block_query_len, heads, heads_dim),
        # keys shape: (N, n_key_blocks, block_key_len, heads, heads_dim)
        # energy: (N, heads, n_query_blocks, block_query_len, block_key_len)

        # del keys, queries
        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)
        # attention shape: (N, heads, num_blocks, query_len, key_len)
        # del energy

        out = torch.einsum("n h a q k , n a k hd -> n a q hd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, num_blocks, query_len, key_len)
        # values shape: (N, num_blocks, block_value_len, heads, heads_dim)
        # out after matrix multiply: (N, num_blocks, block_query_len, heads, head_dim), then
        # we reshape and flatten the (1,2)dimensions as well as (3,4) dimensions.

        if self.use_wout:
            out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return out
    

    
    
############################################################################
######### FOR BLOCK MLP
############################################################################

import sys
sys.path.append("../../")
from SparseNonLinearMixer.sparse_mlp_mixers import BlockLinear

############################################################################
############################################################################

class BlockResMLP(nn.Module):
    def __init__(self, input_dim, layer_dims, actf=nn.ELU):
        super().__init__()
        self.block_dim = layer_dims[0]
        
        assert input_dim%self.block_dim == 0, "Input dim must be even number"
        ### Create a block MLP
        self.mlp = []
        n_blocks = input_dim//layer_dims[0]
        for i in range(len(layer_dims)-1):
            l = BlockLinear(n_blocks, layer_dims[i], layer_dims[i+1])
            a = actf()
            self.mlp.append(l)
            self.mlp.append(a)
            
        self.mlp = self.mlp[:-1]
        self.mlp = nn.Sequential(*self.mlp)
#         self.ln = nn.LayerNorm(self.block_dim)
        
    def forward(self, x):
        bs, dim = x.shape[0], x.shape[1]
        x = x.view(bs, -1, self.block_dim).transpose(0,1)
#         x = self.mlp(self.ln(x)) + x
        x = self.mlp(x) + x
        x = x.transpose(1,0).reshape(bs, -1)
        return x
    
############################################################################
############################################################################

class BlockResMLP_MixerBlock(nn.Module):
    
    def __init__(self, input_dim, block_dim, hidden_layers_ratio=[2], actf=nn.ELU):
        super().__init__()
        
        assert input_dim%block_dim == 0, "Input dim must be even number"
        self.input_dim = input_dim
        self.block_dim = block_dim
        
        def log_base(a, base):
            return np.log(a) / np.log(base)
        
        num_layers = int(np.ceil(log_base(input_dim, base=block_dim)))
        hidden_layers_ratio = [1] + hidden_layers_ratio + [1]
        
        block_layer_dims = [int(a*block_dim) for a in hidden_layers_ratio]
        self.facto_nets = []
        self.gaps = []
        for i in range(num_layers):
            net = BlockResMLP(self.input_dim, block_layer_dims, actf)
            self.facto_nets.append(net)
            
            gap = self.block_dim**i
            if gap*self.block_dim <= self.input_dim:
                self.gaps.append(gap)
            else:
                self.gaps.append(int(np.ceil(self.input_dim/self.block_dim)))
            
        self.facto_nets = nn.ModuleList(self.facto_nets)
            
    def forward(self, x):
        bs = x.shape[0]
        y = x
        for i, fn in enumerate(self.facto_nets):
            gap = self.gaps[i]
            y = y.view(-1, self.block_dim, gap).transpose(2, 1).contiguous().view(bs, -1)
            y = fn(y)
            y = y.view(-1, gap, self.block_dim).transpose(2, 1)

        y = y.contiguous().view(bs, -1)
        return y

#############################################################################
#############################################################################


class Sparse_TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, actf=nn.GELU, embed_block_size:int=None, use_wout=True):
        super().__init__()
        
        self.attention = SelfAttention_Sparse(embed_size, heads, use_wout)
            
        self.norm1 = nn.LayerNorm(embed_size)

        
        if embed_block_size is None or embed_block_size == embed_size:
            self.feed_forward = ResMlpBlock(embed_size, [forward_expansion], actf)
        else:
            self.feed_forward = BlockResMLP_MixerBlock(embed_size, embed_block_size, [forward_expansion], actf)
            
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, block_size):
        attention = self.attention(value, key, query, mask, block_size)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        
        _xs = x.shape
        x = x.view(-1, _xs[-1])
        
        forward = self.feed_forward(x)
#         out = self.dropout(self.norm2(forward + x)) ## skip connection already used
        out = self.dropout(self.norm2(forward))
        return out.view(*_xs)


#############################################################################
#############################################################################


class Mixer_TransformerBlock_Encoder(nn.Module):
    def __init__(self, seq_length, block_size, embed_size, heads, dropout, forward_expansion, actf=nn.GELU, embed_block_size=None, reverse_butterfly=False, use_wout=True):
        super().__init__()
        # This is not just power of 2, but divisible by block size
        # TODO: ^^^
        # assert 2**int(np.log2(block_size)) == block_size, 'Block size must be power of 2'
        # assert 2**int(np.log2(seq_length)) == seq_length, 'Sequence length must be power of 2'
        
        # if embed_block_size is not None:
        #     assert 2**int(np.log2(embed_block_size)) == embed_block_size, 'Embeddings block size must be power of 2'
        assert seq_length%block_size == 0, 'Sequence length must be divisible exactly by block_size'
        
        self.block_size = block_size
        self.seq_len = seq_length
        self.embed_block_size = embed_block_size
        
        def log_base(a, base):
            return np.log(a) / np.log(base)
        
        num_layers = int(np.ceil(log_base(seq_length, base=block_size)))
        self.sparse_transformers = []
        self.gaps = []

        range_val = list(range(num_layers))
        if reverse_butterfly:
            range_val = range_val[::-1]
            
        for i in range_val:            
            tr = Sparse_TransformerBlock(embed_size, heads, dropout, forward_expansion, actf, embed_block_size, use_wout)
            self.sparse_transformers.append(tr)
            ### find which permutation gives valid shape
            gap = self.block_size**i
            if gap*self.block_size <= self.seq_len:
                self.gaps.append(gap)
            else:
                self.gaps.append(int(np.ceil(self.seq_len/self.block_size)))
#                 break
            
        self.sparse_transformers = nn.ModuleList(self.sparse_transformers)
        

    def forward(self, x, mask = None):
        N, seq_len, d_model = x.shape
        ### (N, seq_len, d_model) of the input x
        
        assert seq_len == self.seq_len, 'The sequence length of given input does not match this model'
            
        for i, fn in enumerate(self.sparse_transformers):
            gap = self.gaps[i]
#             print(i, gap)
            x = x.view(N, -1, self.block_size, gap, d_model).transpose(2, 3).contiguous().view(N, seq_len, d_model)
            x = fn(x, x, x, mask, self.block_size)
            x = x.view(N, -1, gap, self.block_size, d_model).transpose(2, 3).contiguous()

        x = x.view(N, seq_len, -1)
        return x
    
    
#############################################################################
#############################################################################





#############################################################################
#############################################################################

### add randomize patches for clear benefit
class Mixer_ViT_Classifier(nn.Module):
    
    def __init__(self, image_dim:tuple, patch_size:tuple, hidden_channel:int, num_blocks:int, num_classes:int, block_seq_size:int, block_mlp_size:int, forward_expansion:float=2.0, pos_emb=False, dropout:float=0.0, randomize_patch:bool=False, reverse_butterfly:bool=False, use_wout:bool=True):
        super().__init__()
        
        self.img_dim = image_dim ### must contain (C, H, W) or (H, W)
        
        ### find patch dim
        d0 = int(image_dim[-2]/patch_size[0])
        d1 = int(image_dim[-1]/patch_size[1])
        assert d0*patch_size[0]==image_dim[-2], "Image must be divisible into patch size"
        assert d1*patch_size[1]==image_dim[-1], "Image must be divisible into patch size"
        
#         self.d0, self.d1 = d0, d1 ### number of patches in each axis
        __patch_size = patch_size[0]*patch_size[1]*image_dim[0] ## number of channels in each patch
    
        ### find channel dim
        channel_size = d0*d1 ## number of patches
        
        ### after the number of channels are changed
        init_dim = __patch_size
        final_dim = hidden_channel
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        #### rescale the patches (patch wise image non preserving transform, unlike bilinear interpolation)
        self.channel_change = nn.Linear(init_dim, final_dim)
        print(f"ViT Mixer : Channels per patch -> Initial:{init_dim} Final:{final_dim}")
        
        
        self.channel_dim = final_dim
        self.patch_dim = channel_size
        
        self.transformer_blocks = []
        
        f = self.get_factors(self.channel_dim)
        print(f)
        fi = np.abs(np.array(f) - np.sqrt(self.channel_dim)).argmin()
        
        _n_heads = f[fi]
        
        ## number of dims per channel -> channel_dim
#         print('Num patches:', self.patch_dim)
        print(f'Sequence len: {self.patch_dim} ; Block size: {block_seq_size}')
        print('Channel dim:', self.channel_dim, 'num heads:',_n_heads)
            
        
        if block_seq_size is None or block_seq_size<2:
            ### Find the block size for sequence:
            block_seq_size = int(2**np.ceil(np.log2(np.sqrt(self.patch_dim))))
            
        print(f'MLP dim: {self.channel_dim} ; Block size: {block_mlp_size}')

        for i in range(num_blocks):
            L = Mixer_TransformerBlock_Encoder(self.patch_dim, block_seq_size, self.channel_dim, _n_heads, dropout, forward_expansion, nn.GELU, block_mlp_size, reverse_butterfly, use_wout)
            self.transformer_blocks.append(L)
        self.transformer_blocks = nn.Sequential(*self.transformer_blocks)
        
        self.linear = nn.Linear(self.patch_dim*self.channel_dim, num_classes)
        
        self.positional_encoding = PositionalEncoding(self.channel_dim, dropout=0)
        if not pos_emb:
            self.positional_encoding = nn.Identity()
            
        self.randomize = None
        if randomize_patch:
            self.randomize = torch.randperm(self.patch_dim)
        # print("Randomize ==",self.randomize)
        
        
    def get_factors(self, n):
        facts = []
        for i in range(2, n+1):
            if n%i == 0:
                facts.append(i)
        return facts
    
    def forward(self, x):
        bs = x.shape[0]
        x = self.unfold(x).swapaxes(-1, -2)
        x = self.channel_change(x)
        x = self.positional_encoding(x)
        ## swap position of patches here
        if self.randomize is not None:
            x = x[..., self.randomize, :]
        x = self.transformer_blocks(x)
        x = self.linear(x.view(bs, -1))
        return x
    
    
#############################################################################
#############################################################################
