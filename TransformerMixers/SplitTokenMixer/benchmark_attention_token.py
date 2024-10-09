import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import transformers_lib_tokenmix as transformer

sys.path.append("../../")
from utils import seed_all, train_model

import torch
import numpy as np
################################################
device = torch.device("cuda:0")
################################################
torch.use_deterministic_algorithms(True) ## remove this for compile
torch.set_float32_matmul_precision('high')
################################################

### Do all experiments in repeat
SAVE_PATH = "./logs"
################################################
################################################

#  -------------------------------------------------
def benchmark_cifar():
    global SAVE_PATH

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default=-1, help = "seed to run benchmark for all models", required = False)
    parser.add_argument("--save_dir", type = str, default="", help = "directory to save the benchmarks on", required = False)

    args = parser.parse_args()

    #------------------------------------------------------

    SEEDS = [147]
    # seedS = [147, 258, 369, 321, 654, 987, 741, 852, 963, 159, 357, 951, 753]

    #------------------------------------------------------
    if args.seed >=0 :
        SEEDS = [args.seed]
    if len(args.save_dir) > 0:
        SAVE_PATH = args.save_dir 
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    #------------------------------------------------------


    # EPOCHS = 1
    EPOCHS = 300
    LR = 0.0001

    #------------------------------------------------------
    ### CONFIGS TUPLE
    ## (DS, num_layers, patch_size)
    configs = [
        # ('c10', 8, 4),
        ('c10', 4, 4),
        # ('c10', 4, 2),
        # ('c10', 4, 1),
        # ('c100', 4, 4),
        # ('c100', 4, 2),
    ]
    patch_expansion_dict = {16:1024, 8:256, 4:128, 2:64, 1:64}
    imsize = (3, 32, 32)

    for seed in SEEDS:
        for config in configs:
            for heads_per_parallel_attn in [1]: ### for parallel attention
                global patch_size
                DS, num_layers, patch_size = config
                #------------------------------------------------------
                num_cls = 10
                if DS=='c100': num_cls = 100
                #------------------------------------------------------
                embedding_dim = patch_expansion_dict[patch_size]
                block_mlp_dim = embedding_dim

                #------------------------------------------------------
                # block_seq_len = (imsize[-1]*imsize[-2])//(patch_size*patch_size)
                num_blocks = num_layers
                #------------------------------------------------------
                seed_all(seed)
                if heads_per_parallel_attn is not None:
                    #------------------------------------------------------
                    model = transformer.Mixer_ViT_parallelAttention_Classifier(
                                                            imsize,
                                                            (patch_size, patch_size),
                                                            embedding_dim,
                                                            num_blocks,
                                                            num_cls,
                                                            heads_per_parallel_attn,
                                                            )

                    #------------------------------------------------------
                    ### we seem to get better accuracy without Positional Encoding (PE) -> named as nPE in experiments.
                    model_name = f"ViT_nPE_parallelAtt{heads_per_parallel_attn}_{DS}_patch{patch_size}_l{num_layers}_s{seed}"
                else:
                    block_seq_len = (imsize[-1] * imsize[-2]) // (patch_size * patch_size)
                    model = transformer.Mixer_ViT_Classifier(imsize,
                                                             (patch_size, patch_size),
                                                             embedding_dim,
                                                             num_blocks,
                                                             num_cls,
                                                             block_seq_size=block_seq_len,
                                                             block_mlp_size=embedding_dim,
                                                             use_wout=False
                                                             )
                    model_name = f"ViT_nPE_att_noWout_{DS}_patch{patch_size}_l{num_layers}_s{seed}"

                print(model)
                #------------------------------------------------------
                train_model(model, LR, model_name, DS, EPOCHS)
                #------------------------------------------------------
            # ------------------------------------------------------
        #------------------------------------------------------
    #------------------------------------------------------

################################################

benchmark_cifar()

################################################
################################################
################################################



