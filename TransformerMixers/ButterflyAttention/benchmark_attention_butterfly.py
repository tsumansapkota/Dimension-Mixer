import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import transformers_lib_butterfly as transformer

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
    # SEEDS = [147, 258, 369]
    # seedS = [147, 258, 369, 321, 654, 987, 741, 852, 963, 159, 357, 951, 753]

    #------------------------------------------------------
    if args.seed >=0 :
        SEEDS = [args.seed]
    if len(args.save_dir) > 0:
        SAVE_PATH = args.save_dir 
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    #------------------------------------------------------


    # EPOCHS = 2
    EPOCHS = 300
    LR = 0.0001

    #------------------------------------------------------
    ### CONFIGS TUPLE
    ## (DS, num_layers, patch_size)
    configs = [
        #('c10', 8, 4),
        ('c10', 4, 4),
        #('c10', 4, 2),
        #('c10', 4, 1),
        #('c100', 4, 4),
        #('c100', 4, 2),
    ]
    patch_expansion_dict = {16:1024, 8:256, 4:128, 2:64, 1:64}
    imsize = (3, 32, 32)

    for seed in SEEDS:
        for config in configs:
            global patch_size
            DS, num_layers, patch_size = config
            #------------------------------------------------------
            num_cls = 10
            if DS=='c100': num_cls = 100
            #------------------------------------------------------
            for butterfly_att in [False, True]:
                if butterfly_att:
                    randomize_patch = [False, True]
                else:
                    randomize_patch = [False]

                for rand_patch in randomize_patch:
                    reverse_butterfly = [False]
                    # if randomize_patch is True: ## uncomment to use reverse_butterfly, we do not find difference
                    #     reverse_butterfly = [True, False]
                        
                    for rev_butterfly in reverse_butterfly:

                        #------------------------------------------------------
                        if butterfly_att: ## using 2 mixing blocks
                            assert num_layers%2 == 0, 'number of blocks on sparse transformer is (x2)/2 hence it must be even'
                            num_blocks = num_layers//2
                        else:
                            num_blocks = num_layers

                        #------------------------------------------------------
                        embedding_dim = patch_expansion_dict[patch_size]
                        block_mlp_dim = embedding_dim

                        #------------------------------------------------------
                        block_seq_len = (imsize[-1]*imsize[-2])//(patch_size*patch_size)
                        if butterfly_att:
                            block_seq_len = int(2**np.ceil(np.log2(np.sqrt(block_seq_len))))

                        #------------------------------------------------------
                        seed_all(seed)
                        #------------------------------------------------------
                        model = transformer.Mixer_ViT_Classifier(
                                                                imsize, 
                                                                (patch_size, patch_size), 
                                                                embedding_dim, 
                                                                num_blocks, 
                                                                num_cls, 
                                                                block_seq_len, 
                                                                block_mlp_dim,
                                                                pos_emb=False,
                                                                randomize_patch=rand_patch,
                                                                )

                        #------------------------------------------------------
                        _a, _b, _c = 'att', '', ''
                        if butterfly_att: _a = 'sAtt'
                        if rand_patch: _b = '_rand'
                        if rev_butterfly: _c = '_revB'

                        ### we seem to get better accuracy without Positional Encoding (PE) -> named as nPE in experiments.
                        model_name = f"ViT_nPE_{DS}_patch{patch_size}_l{num_layers}_{_a}{_b}{_c}_s{seed}"
                        #------------------------------------------------------
                        train_model(model, LR, model_name, DS, EPOCHS)
                    #------------------------------------------------------
                #------------------------------------------------------
            #------------------------------------------------------
        #------------------------------------------------------
    #------------------------------------------------------

################################################

benchmark_cifar()

################################################
################################################
################################################



