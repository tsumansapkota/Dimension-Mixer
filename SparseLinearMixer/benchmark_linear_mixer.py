import torch
import torch.nn as nn
import numpy as np
import os

import sparse_linear_lib as sll

import pandas as pd
import time

####################################################################
device = torch.device("cuda:0")


####################################################################

def train_model(model, optimizer, X, A):
    for i in range(TRAIN_STEPS):
        out = model(X)
        loss = mse(out, A)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%1000 == 0:
            print(f"The MSE loss is : {float(mse(out,A))}")
            
    with torch.no_grad():
        start = time.time()
        out = model(X)
        tt = time.time()-start
    return out, tt

def get_svd_output(A, n_comp):
    U, S, V = torch.svd(A)

    with torch.no_grad():
        _U = U[:, :n_comp]
        _S = S[:n_comp]
        _V = V[:, :n_comp]
        
        start = time.time()
        out = torch.mm(torch.mm(_U, torch.diag(_S)), _V.t())
        tt = time.time()-start
    
    return out, tt

def save_stats(df, out, A, method, seed, nparam, tim, filename):
    global SAVE_PATH
    diff = (out.data-A).abs()
    
    mean, std = float(diff.mean()), float(diff.std())
    err = float(mse(out, A))

    # df = df.append({"method":method, "seed":seed, "MSE":err,
    #                 "MAE":mean, "std-MAE":std, "params":nparam, "time":tim},
    #                ignore_index=True)
    new_row = {"method":method, "seed":seed, "MSE":err,
                    "MAE":mean, "std-MAE":std, "params":nparam, "time":tim}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df_ = df.copy()
    df.to_csv(f"{SAVE_PATH}/{file_name}.csv")
    
    print(f"Saving... file:{file_name} method:{method}")
    return df_

####################################################################

# df = pd.DataFrame(columns=['method'])

####################################################################

class Add_PairLinears(nn.Module):
    
    def __init__(self, input_dim, num_adds):
        super().__init__()
        self.pair_mixers = []
        self.perm_indices = []
        for i in range(num_adds):
            # m = sll.PairLinear_MixerBlock(input_dim, input_dim)
            m = sll.BlockLinear_MixerBlock(input_dim, 2)        
            self.pair_mixers.append(m)
            if i > 0:
                rm = torch.randperm(input_dim)
                self.perm_indices.append(rm)
                
        self.pair_mixers = nn.ModuleList(self.pair_mixers)
        
    def forward(self, x):
        y = torch.zeros_like(x)
        for i, m in enumerate(self.pair_mixers):
            if i > 0:
                _x = x[:, self.perm_indices[i-1]]
            else:
                _x = x
                
            y += m(_x)
        return y

####################################################################

class Stack_PairLinears(nn.Module):
    
    def __init__(self, input_dim, num_adds):
        super().__init__()
        self.pair_mixers = []
        self.perm_indices = []
        for i in range(num_adds):
            # m = sll.PairLinear_MixerBlock(input_dim, input_dim)
            m = sll.BlockLinear_MixerBlock(input_dim, 2)        
            self.pair_mixers.append(m)
            if i > 0:
                rm = torch.randperm(input_dim)
                self.perm_indices.append(rm)
                
        self.pair_mixers = nn.ModuleList(self.pair_mixers)
        
    def forward(self, x):
        for i, m in enumerate(self.pair_mixers):
            if i == 0:
                x = m(x)
            else:
                x = m(x[:, self.perm_indices[i-1]])
        return x

####################################################################
def log_base(a, base):
    return np.log(a) / np.log(base)

####################################################################
####################################################################

import warnings
warnings.filterwarnings('ignore')

'''
FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
  df = df.append({"method":method, "seed":seed, "mse":err,
'''
####################################################################

## Pair Linear approximation

Ns = [16, 64, 256, 1024, 4096] #, 16384]


mse = nn.MSELoss()
TRAIN_STEPS = 20000 #default


SAVE_PATH = "./logs/"
####################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default=-1, help = "seed to run benchmark for all models", required = False)
parser.add_argument("--save_dir", type = str, default="", help = "directory to save the benchmarks on", required = False)

args = parser.parse_args()

SEEDS = [147, 258, 369, 321, 654, 987, 741, 852, 963, 159, 357, 951, 753]

#------------------------------------------------------
if args.seed >=0 :
    SEEDS = [args.seed]
if len(args.save_dir) > 0:
    SAVE_PATH = args.save_dir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
#------------------------------------------------------

A = None
for SEED in SEEDS:
    for N in Ns:
    #     torch.cuda.empty_cache()
        X = torch.eye(N).to(device)
        
        df = pd.DataFrame(columns=['method', 'seed', 'MSE', "MAE", "std-MAE", 'params', 'time'])
        file_name = f'record_err_{N}_s{SEED}'
        
        print()
        print(f"Experiment N={N} SEED={SEED}")
        
        torch.manual_seed(SEED)
        
        del A
        
        A = torch.rand(N, N).to(device)*2-1
        ### For each method compute the stats
        
        #####################################################
        ##### First 2x2 factorization
        # model = sll.PairLinear_MixerBlock(N, N).to(device)    
        model = sll.BlockLinear_MixerBlock(N, 2).to(device)        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        
        #### Train model
        out, tim = train_model(model, optimizer, X, A)
        _n_params = sum(p.numel() for p in model.parameters())
        df = save_stats(df, out, A, 'pair', SEED, _n_params, tim, file_name)
        #####################################################
        del model, optimizer
        
        #####################################################
        #### sqrt(N)/2 
        _m = int(np.ceil(np.sqrt(N)/2))
        model = sll.BlockLinear_MixerBlock(N, _m).to(device)        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        
        #### Train model
        out, tim = train_model(model, optimizer, X, A)
        _n_params = sum(p.numel() for p in model.parameters())
        df = save_stats(df, out, A, 'block-sqrt-half', SEED, _n_params, tim, file_name)
        #####################################################
        del model, optimizer
        
        
        #####################################################
        ##### Second sqrt(N) factorization
        _m = int(np.ceil(np.sqrt(N)))
        model = sll.BlockLinear_MixerBlock(N, _m).to(device)        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        
        #### Train model
        out, tim = train_model(model, optimizer, X, A)
        _n_params = sum(p.numel() for p in model.parameters())
        df = save_stats(df, out, A, 'block-sqrt', SEED, _n_params, tim, file_name)
        #####################################################
        del model, optimizer
        
        
        #####################################################
        ##### Low Rank or SVD
        _m = int(np.ceil(_n_params/(N*2)))
        out, tim = get_svd_output(A, _m)

        n_params = N*_m*2
        df = save_stats(df, out, A, 'lowR-same-param', SEED, n_params, tim, file_name)
        #####################################################
        
        #####################################################
        _m = N // 2
        out, tim = get_svd_output(A, _m)

        n_params = N*_m*2
        df = save_stats(df, out, A, 'lowR-half', SEED, n_params, tim, file_name)
        #####################################################
        
        #####################################################
#         torch.cuda.empty_cache()
        if N > 1024: continue ## To avoid OOM error

        ##### Pair Linear models parallel addition
        _m = int(np.ceil(np.log2(N)))
        model = Add_PairLinears(N, _m).to(device)        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        
        #### Train model
        out, tim = train_model(model, optimizer, X, A)
        _n_params = sum(p.numel() for p in model.parameters())
        df = save_stats(df, out, A, 'pair-Add', SEED, _n_params, tim, file_name)
        #####################################################
        del model, optimizer
        
        #####################################################
        
        ##### Pair Linear models sequential composition
        _m = int(np.ceil(np.log2(N)))
        model = Stack_PairLinears(N, _m).to(device)        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        
        #### Train model
        out, tim = train_model(model, optimizer, X, A)
        _n_params = sum(p.numel() for p in model.parameters())
        df = save_stats(df, out, A, 'pair-Seq', SEED, _n_params, tim, file_name)
        #####################################################
        del model, optimizer
#         torch.cuda.empty_cache()