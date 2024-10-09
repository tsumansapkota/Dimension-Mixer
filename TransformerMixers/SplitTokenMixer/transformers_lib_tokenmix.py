import torch
import torch.nn as nn
import numpy as np

#############################################################################
#############################################################################

############################################################################
######### FOR BLOCK MLP
############################################################################

import sys
sys.path.append("../")
from ButterflyAttention.transformers_lib_butterfly import BlockResMLP, ResMlpBlock, BlockLinear, PositionalEncoding, Mixer_ViT_Classifier

#############################################################################
#############################################################################

class ParallelSelfAttention(nn.Module):
    def __init__(self, embed_size, embed_block_size, heads, use_wout = True):
        super().__init__()
        self.embed_size = embed_size
        self.embed_block_size = embed_block_size
        self.num_blocks = embed_size // embed_block_size

        self.heads = heads
        self.head_dim = embed_block_size // heads

        self.use_wout = use_wout

        assert (
                self.head_dim * heads == embed_block_size
        ), "Embedding size needs to be divisible by heads"

        assert (
                embed_size % self.embed_block_size == 0
        ), "Embedding block size should exactly divide embedding size"

        #         self.values = nn.Linear(embed_size, embed_size)
        self.values = BlockLinear(self.num_blocks, embed_block_size, embed_block_size)
        #         self.keys = nn.Linear(embed_size, embed_size)
        self.keys = BlockLinear(self.num_blocks, embed_block_size, embed_block_size)
        #         self.queries = nn.Linear(embed_size, embed_size)
        self.queries = BlockLinear(self.num_blocks, embed_block_size, embed_block_size)

        if self.use_wout:
            # self.fc_out = nn.Linear(embed_size, embed_size)
            self.fc_out = BlockLinear(self.num_blocks, embed_block_size, embed_block_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.view(-1, self.num_blocks, self.embed_block_size).transpose(0, 1)
        values = self.values(values).transpose(0, 1)  # (N*value_len, num_block, embed_block_size)
        #         print(values.shape)

        keys = keys.view(-1, self.num_blocks, self.embed_block_size).transpose(0, 1)
        keys = self.keys(keys).transpose(0, 1)  # (N*key_len, num_block, embed_block_size)

        query = query.view(-1, self.num_blocks, self.embed_block_size).transpose(0, 1)
        queries = self.queries(query).transpose(0, 1)  # (N*query_len, num_block, embed_block_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.num_blocks, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_blocks, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_blocks, self.heads, self.head_dim)

        #         print(keys.shape, queries.shape, values.shape)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqbhd,nkbhd->nbhqk", [queries, keys])
        # queries shape: (N, query_len, num_blocks, heads, heads_dim),
        # keys shape: (N, key_len, num_blocks, heads, heads_dim)
        # energy: (N, num_blocks, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        # attention shape: (N, num_blocks, heads, query_len, key_len)

        out = torch.einsum("nbhql,nlbhd->nqbhd", [attention, values]).reshape(
            N * query_len, self.num_blocks, self.heads * self.head_dim
        )
        # attention shape: (N, heads, num_blocks, query_len, key_len)
        # values shape: (N, value_len, num_blocks, heads, heads_dim)
        # out after matrix multiply: (N, query_len, num_blocks, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        #         print(out.shape)

        #         print(out.shape)


        if self.use_wout:
            # out = torch.einsum("n h a q k , n a k hd -> n a q hd", [attention, values]).reshape(
            #     N, query_len, self.heads * self.head_dim
            # )
            ############################################
            out = self.fc_out(out.transpose(0, 1))  ## takes in NumBlocks, BS, block_dim
            # Linear layer doesn't modify the shape, final shape will be
            # (N, query_len, embed_size)
            # (N, query_len, num_blocks, embed_block_size)
            #         print(out.shape)
            out = out.transpose(0, 1).contiguous()
            #         print(out.shape)
            ############################################
            pass
        out = out.view(N, query_len, -1)
        #         print(out.shape)
        return out

#############################################################################
#############################################################################

#############################################################################
#############################################################################


class TokenSparse_TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, actf=nn.GELU, embed_block_size: int = None,
                 block_mlp=False):
        super().__init__()

        num_blocks = embed_size // embed_block_size

        self.embed_block_size = embed_block_size
        self.attention = ParallelSelfAttention(embed_size, embed_block_size, heads // num_blocks, use_wout=(heads == embed_size//embed_block_size))

        self.norm1 = nn.LayerNorm(embed_size)

        if not block_mlp:
            self.feed_forward = ResMlpBlock(embed_size, [forward_expansion], actf)
        else:
            self.feed_forward = BlockResMLP(embed_size,
                                         [embed_block_size, int(embed_block_size * forward_expansion),
                                          embed_block_size],
                                         actf)

        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))

        _xs = x.shape
        x = x.view(-1, _xs[-1])

        forward = self.feed_forward(x)
        #         out = self.dropout(self.norm2(forward + x))
        out = self.dropout(self.norm2(forward))  ## because resmlp or blockmlp is used (already have residual)
        return out.view(*_xs)

#############################################################################
#############################################################################

class Mixer_TokenSparse_TransformerBlock_Encoder(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, actf=nn.GELU, embed_block_size=None,
                 block_mlp=False):
        super().__init__()
        #         assert 2**int(np.log2(block_size)) == block_size, 'Block size must be power of 2'
        #         assert 2**int(np.log2(seq_length)) == seq_length, 'Sequence length must be power of 2'

        if embed_block_size is not None:
            assert 2 ** int(np.log2(embed_block_size)) == embed_block_size, 'Embeddings block size must be power of 2'
            assert embed_size % embed_block_size == 0, 'Embedding dim must be divisible exactly by embed_block_size'

        self.embed_size = embed_size
        self.embed_block_size = embed_block_size

        def log_base(a, base):
            return np.log(a) / np.log(base)

        num_layers = int(np.ceil(log_base(embed_size, base=embed_block_size)))
        self.sparse_transformers = []
        # self.gaps = []
        for i in range(num_layers):
            tr = TokenSparse_TransformerBlock(embed_size, heads, dropout,
                                              forward_expansion, actf, embed_block_size, block_mlp)
            self.sparse_transformers.append(tr)

            ### find which permutation gives valid shape
            # gap = self.embed_block_size ** i
            # if gap * self.embed_block_size <= embed_size:
            #     self.gaps.append(gap)
            # else:
            #     self.gaps.append(int(np.ceil(self.embed_size / self.embed_block_size)))

        self.sparse_transformers = nn.ModuleList(self.sparse_transformers)

    def forward(self, x, mask=None):
        N, seq_len, d_model = x.shape
        ### (N, seq_len, d_model) of the input x

        assert d_model == self.embed_size, 'The embeddig dim of given input does not match this model'

        for i, fn in enumerate(self.sparse_transformers): ## permutation makes less impact as MLP mixes the tokens.
            # gap = self.gaps[i]
            #             print(i, gap)
            # x = x.view(N, seq_len, -1, self.embed_block_size, gap).transpose(3, 4).contiguous().view(N, seq_len,
            #                                                                                          d_model)
            x = fn(x, x, x, mask)
            # x = x.view(N, seq_len, -1, gap, self.embed_block_size).transpose(3, 4).contiguous()

        # x = x.view(N, seq_len, -1)
        return x

#############################################################################
#############################################################################

class Mixer_ViT_parallelAttention_Classifier(nn.Module):

    def __init__(self, image_dim: tuple, patch_size: tuple, embedding_dim: int, num_blocks: int, num_classes: int, heads_per_mhsa:int):
        super().__init__()

        self.img_dim = image_dim  ### must contain (C, H, W) or (H, W)

        ### find patch dim
        d0 = int(image_dim[-2] / patch_size[0])
        d1 = int(image_dim[-1] / patch_size[1])
        assert d0 * patch_size[0] == image_dim[-2], "Image must be divisible into patch size"
        assert d1 * patch_size[1] == image_dim[-1], "Image must be divisible into patch size"
        #         self.d0, self.d1 = d0, d1 ### number of patches in each axis
        __patch_size = patch_size[0] * patch_size[1] * image_dim[0]  ## number of channels in each patch

        ### find channel dim
        channel_size = d0 * d1  ## number of patches

        ### after the number of channels are changed
        init_dim = __patch_size
        # final_dim = int(__patch_size * hidden_expansion / 2) * 2
        final_dim = embedding_dim
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

        print(self.channel_dim, _n_heads)

        # emb_blk_sz = 2 ** int(np.ceil(np.log2(self.channel_dim) / 2))
        assert _n_heads % heads_per_mhsa == 0, f"using {heads_per_mhsa} head per block of parallel attention, so heads = {_n_heads} must be divisible by {heads_per_mhsa}"
        emb_blk_sz = self.channel_dim//(_n_heads//heads_per_mhsa)

        def log_base(a, base): return np.log(a) / np.log(base)
        num_layers = int(np.ceil(log_base(self.channel_dim, base=emb_blk_sz)))
        for i in range(num_blocks // num_layers):
            L = Mixer_TokenSparse_TransformerBlock_Encoder(self.channel_dim, _n_heads, 0, 2,
                                                           embed_block_size=emb_blk_sz, block_mlp=False)
            self.transformer_blocks.append(L)
        self.transformer_blocks = nn.Sequential(*self.transformer_blocks)

        self.linear = nn.Linear(self.patch_dim * self.channel_dim, num_classes)
        # self.positional_encoding = PositionalEncoding(self.channel_dim, dropout=0)
        # self.positional_encoding = PositionalEncoding(self.channel_dim, dropout=0)
        # if not pos_emb:
        #     self.positional_encoding = nn.Identity()

    def forward(self, x):
        bs = x.shape[0]
        x = self.unfold(x).swapaxes(-1, -2)
        x = self.channel_change(x)
        #         x = self.positional_encoding(x)
        x = self.transformer_blocks(x)
        x = self.linear(x.view(bs, -1))
        return x

    def get_factors(self, n):
        facts = []
        for i in range(2, n + 1):
            if n % i == 0:
                facts.append(i)
        return facts



#############################################################################
#############################################################################

### Parallel Self Attention --> mixed by MLP layer
### Parallel Self Attention with 1-head per parallel block --> mixed by MLP layer

#############################################################################