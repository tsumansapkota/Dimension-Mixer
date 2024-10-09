import torch
import torch.nn as nn
import numpy as np


############################################################################
#
class BlockLinear_conv(nn.Module):
    def __init__(self, num_blocks, input_block_dim, output_block_dim, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(input_block_dim * num_blocks, output_block_dim * num_blocks,
                              kernel_size=1, groups=num_blocks, bias=bias)

    def forward(self, x):
        nblocks, bs, dim = x.shape[0], x.shape[1], x.shape[2]
        x = x.transpose(0, 1).reshape(bs, -1, 1, 1)
        x = self.conv(x).reshape(bs, nblocks, -1).transpose(0, 1)
        return x

    def __repr__(self):
        S = f'BlockLinear_conv: {list(self.conv.weight.shape)}'
        return S


############################################################################

class BlockLinear(nn.Module):
    def __init__(self, num_blocks, input_block_dim, output_block_dim, bias=True):
        super().__init__()
        self.weight = torch.randn(num_blocks, input_block_dim, output_block_dim)

        self.weight = nn.Parameter(self.weight)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.weight.shape[0], 1, output_block_dim))

    def forward(self, x):
        # nblocks, bs, dim = x.shape[0], x.shape[1], x.shape[2]
        x = torch.bmm(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def __repr__(self):
        S = f'BlockLinear: {list(self.weight.shape)}'
        return S

BlockLinear = BlockLinear_conv ### this helps measure MACS/FLOPS easy
############################################################################

class MlpBLock(nn.Module):

    def __init__(self, input_dim, hidden_layers_ratio=[2], actf=nn.GELU):
        super().__init__()
        self.input_dim = input_dim
        #### convert hidden layers ratio to list if integer is inputted
        if isinstance(hidden_layers_ratio, int):
            hidden_layers_ratio = [hidden_layers_ratio]

        self.hlr = [1] + hidden_layers_ratio + [1]

        self.mlp = []
        ### for 1 hidden layer, we iterate 2 times
        for h in range(len(self.hlr) - 1):
            i, o = int(self.hlr[h] * self.input_dim), \
                int(self.hlr[h + 1] * self.input_dim)
            self.mlp.append(nn.Linear(i, o))
            self.mlp.append(actf())
        self.mlp = self.mlp[:-1]

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)

############################################################################

class MLP_SparseLinear_Monarch_Deform(nn.Module):

    def __init__(self, input_dim, block_dim, hidden_expansion=2, actf=nn.ELU):
        super().__init__()
        # assert input_dim % block_dim == 0, "Input dim must be divisible by block dim"
        assert np.sqrt(input_dim) == block_dim, "Input dim must be square of block dim"
        assert hidden_expansion >= 1

        self.input_dim = input_dim
        self.block_dim = block_dim
        self.hidden_expansion = hidden_expansion

        self.hidden_dim = input_dim * hidden_expansion

        def log_base(a, base):
            return np.log(a) / np.log(base)

        num_layers = int(np.ceil(log_base(input_dim, base=block_dim)))
        assert num_layers == 2, "Num layers > 2 does not contribute to monarch"

        self.linear0_0 = BlockLinear(self.input_dim // block_dim, block_dim, block_dim * hidden_expansion,
                                         bias=False)
        self.linear0_1 = BlockLinear(self.hidden_dim // block_dim, block_dim, block_dim, bias=True)
        self.stride0 = block_dim * hidden_expansion

        self.actf = actf()
        self.linear1_0 = BlockLinear(self.hidden_dim // block_dim, block_dim, block_dim, bias=False)
        self.linear1_1 = BlockLinear(self.input_dim // block_dim, block_dim * hidden_expansion, block_dim,
                                         bias=True)

    def forward(self, x):
        ## Say shape of x is [BS, 121] > hidden expansion 2

        bs = x.shape[0]  ## BS, input_dim
        y = x

        y = y.view(bs, -1, self.block_dim)  ## BS, num_blocks, block_dim ; [bs, 11, 11]
        y = y.transpose(0, 1).contiguous()  ## num_blocks, BS, block_dim ; [11, bs, 11]
        y = self.linear0_0(y)  ## num_blocks, BS, block_dim*hidden_expansion ; [11, bs, 22]
        y = y.transpose(0, 1).contiguous()  ## BS, num_blocks, block_dim*hidden_expansion ; [bs, 11, 22]
        y = y.view(bs, -1)  ## BS, hidden_dim ; [bs, 242]

        y = y.view(bs, self.block_dim, self.stride0).permute(2, 0,
                                                             1).contiguous()  ## num_blocks, BS, block_dim; [22, bs, 11]
        y = self.linear0_1(y)  ## num_blocks, BS, block_dim ; [22, bs, 11]
        y = y.transpose(0, 1).contiguous()  ## BS, num_blocks, block_dim ; [bs, 22, 11]
        y = y.view(bs, -1)  ## BS, hidden_dim ; [bs, 242]

        ### First linear complete
        y = self.actf(y)

        y = y.view(bs, -1, self.block_dim)  ## BS, num_blocks, block_dim ; [bs, 22, 11]
        y = y.transpose(0, 1).contiguous()  ## num_blocks, BS, block_dim ; [22, bs, 11]
        y = self.linear1_0(y)  ## num_blocks, BS, block_dim*hidden_expansion ; [22, bs, 11]
        y = y.transpose(0, 1).contiguous()  ## BS, num_blocks, block_dim*hidden_expansion ; [bs, 22, 11]
        y = y.view(bs, -1)  ## BS, hidden_dim ; [bs, 242]

        y = y.view(bs, self.stride0, self.block_dim).permute(2, 0,
                                                             1).contiguous()  ## num_blocks, BS, block_dim; [11, bs, 22]
        y = self.linear1_1(y)  ## num_blocks, BS, block_dim ; [11, bs, 11]
        y = y.transpose(0, 1).contiguous()  ## BS, num_blocks, block_dim ; [bs, 11, 11]
        y = y.view(bs, -1)  ## BS, hidden_dim ; [bs, 121]

        return y

############################################################################

class BlockMLP(nn.Module):
    def __init__(self, input_dim, layer_dims, actf=nn.GELU):
        super().__init__()
        self.block_dim = layer_dims[0]

        assert input_dim % self.block_dim == 0, "Input dim must be divisible by block dim"
        ### Create a block MLP
        self.mlp = []
        n_blocks = input_dim // layer_dims[0]
        for i in range(len(layer_dims) - 1):
            l = BlockLinear(n_blocks, layer_dims[i], layer_dims[i + 1])
            a = actf()
            self.mlp.append(l)
            self.mlp.append(a)
        self.mlp = self.mlp[:-1]
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        bs, dim = x.shape[0], x.shape[1]
        x = x.view(bs, -1, self.block_dim).transpose(0, 1)
        x = self.mlp(x)
        x = x.transpose(1, 0).reshape(bs, -1)
        return x


############################################################################

class BlockMLP_MixerBlock(nn.Module):

    def __init__(self, input_dim, block_dim, hidden_layers_ratio=[2], actf=nn.GELU):
        super().__init__()

        assert input_dim % block_dim == 0, "Input dim must be divisible by block dim"
        self.input_dim = input_dim
        self.block_dim = block_dim

        def log_base(a, base):
            return np.log(a) / np.log(base)

        num_layers = int(np.ceil(log_base(input_dim, base=block_dim)))
        hidden_layers_ratio = [1] + hidden_layers_ratio + [1]

        block_layer_dims = [int(a * block_dim) for a in hidden_layers_ratio]
        self.facto_nets = []
        for i in range(num_layers):
            net = BlockMLP(self.input_dim, block_layer_dims, actf)
            self.facto_nets.append(net)

        self.facto_nets = nn.ModuleList(self.facto_nets)

    def forward(self, x):
        bs = x.shape[0]
        y = x
        for i, fn in enumerate(self.facto_nets):
            y = y.view(-1, self.block_dim, self.block_dim ** i).permute(0, 2, 1).contiguous().view(bs, -1)
            y = fn(y)
            y = y.view(-1, self.block_dim ** i, self.block_dim).permute(0, 2, 1).contiguous()

        y = y.view(bs, -1)
        return y

############################################################################
############################################################################

class MixerBlock(nn.Module):

    def __init__(self, patch_dim, channel_dim, patch_mixing="dense", channel_mixing="dense", hidden_expansion = 2):
        super().__init__()
        self.hidden_expansion = hidden_expansion

        self.valid_functions = ["dense", "sparse_linear", "sparse_mlp"]
        assert patch_mixing in self.valid_functions
        assert channel_mixing in self.valid_functions

        self.patch_dim = patch_dim
        self.channel_dim = channel_dim

        self.ln0 = nn.LayerNorm(channel_dim)
        self.mlp_patch = self.get_mlp(patch_dim, patch_mixing)

        self.ln1 = nn.LayerNorm(channel_dim)
        self.mlp_channel = self.get_mlp(channel_dim, channel_mixing)

    def get_mlp(self, dim, mixing_function):
        block_dim = int(np.sqrt(dim))
        assert block_dim ** 2 == dim, "Sparsifying dimension must be a square number"

        if mixing_function == self.valid_functions[0]:
            mlp = MlpBLock(dim, [self.hidden_expansion])
        elif mixing_function == self.valid_functions[1]:
            mlp = MLP_SparseLinear_Monarch_Deform(dim, block_dim, self.hidden_expansion)

        elif mixing_function == self.valid_functions[2]:
            mlp = BlockMLP_MixerBlock(dim, block_dim, [self.hidden_expansion])
        return mlp

    def forward(self, x):
        ## x has shape-> N, nP, nC/hidden_dims; C=Channel, P=Patch

        ######## !!!! Can use same mixer on shape of -> N, C, P;

        #### mix per patch
        y = self.ln0(x)  ### per channel layer normalization ??
        y = torch.swapaxes(y, -1, -2).contiguous()

        y = y.view(-1, self.patch_dim)
        y = self.mlp_patch(y)
        y = y.view(-1, self.channel_dim, self.patch_dim)

        y = torch.swapaxes(y, -1, -2)
        x = x + y

        #### mix per channel
        y = self.ln1(x)
        y = y.view(-1, self.channel_dim)
        y = self.mlp_channel(y)
        y = y.view(-1, self.patch_dim, self.channel_dim)

        x = x + y
        return x

############################################################################
class MlpMixer(nn.Module):

    def __init__(self, image_dim: tuple, patch_size: tuple, hidden_expansion: float, num_blocks: int, num_classes: int,
                 patch_mixing: str, channel_mixing: str, mlp_expansion:int):
        super().__init__()

        self.img_dim = image_dim  ### must contain (C, H, W) or (H, W)
        self.scaler = nn.UpsamplingBilinear2d(size=(self.img_dim[-2], self.img_dim[-1]))

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
        #         final_dim = int(patch_size[0]*patch_size[1]*hidden_expansion)
        final_dim = int(init_dim * hidden_expansion)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        #### rescale the patches (patch wise image non preserving transform, unlike bilinear interpolation)
        self.channel_change = nn.Linear(init_dim, final_dim)
        print(f"MLP Mixer : Channes per patch -> Initial:{init_dim} Final:{final_dim}")

        self.channel_dim = final_dim
        self.patch_dim = channel_size

        self.mixer_blocks = []
        for i in range(num_blocks):
            self.mixer_blocks.append(MixerBlock(self.patch_dim, self.channel_dim, patch_mixing, channel_mixing, mlp_expansion))
        self.mixer_blocks = nn.Sequential(*self.mixer_blocks)

        self.linear = nn.Linear(self.patch_dim * self.channel_dim, num_classes)

    def forward(self, x):
        bs = x.shape[0]
        x = self.scaler(x)
        x = self.unfold(x).swapaxes(-1, -2)
        x = self.channel_change(x)
        x = self.mixer_blocks(x)
        x = self.linear(x.view(bs, -1))
        return x

############################################################################
