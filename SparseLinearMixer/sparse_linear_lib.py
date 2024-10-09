import torch
import torch.nn as nn
import numpy as np

############################################################################
############################################################################

class BlockWeight(nn.Module):
    def __init__(self, input_dim, block_dim):
        super().__init__()
        self.block_dim = block_dim
        
        assert input_dim%block_dim == 0, "Input dim must be divisible by block dim"
        self.weight = torch.eye(block_dim).unsqueeze(0).repeat_interleave(input_dim//block_dim, dim=0)
        self.weight = nn.Parameter(self.weight)
        
    def forward(self, x):
        bs, dim = x.shape[0], x.shape[1]
        x = x.view(bs, -1, self.block_dim).transpose(0,1)
        x = torch.bmm(x, self.weight)
        x = x.transpose(1,0).reshape(bs, -1)
        return x
    
    
class BlockLinear_MixerBlock(nn.Module):
    
    def __init__(self, input_dim, block_dim):
        super().__init__()
        
        assert input_dim%block_dim == 0, "Input dim must be divisible by block dim"
        
        self.input_dim = input_dim
        self.block_dim = block_dim
        
        def log_base(a, base):
            return np.log(a) / np.log(base)
        
        num_layers = int(np.ceil(log_base(input_dim, base=block_dim)))
            
        self.facto_nets = []
        self.gaps = []
        for i in range(num_layers):
            net = BlockWeight(self.input_dim, block_dim)
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
  
            y = y.view(bs, -1, self.block_dim, gap).transpose(2, 3).contiguous().view(bs, -1)
            y = fn(y)
            y = y.view(bs, -1, gap, self.block_dim).transpose(2, 3).contiguous()

        y = y.view(bs, -1)
        return y


############################################################################
############################################################################