import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from sparse_mlp_mixers import MlpMixer

sys.path.append("../")
from utils import seed_all, train_model

import torch
################################################
device = torch.device("cuda:0")
################################################
torch.use_deterministic_algorithms(True) ## remove this for compile
torch.set_float32_matmul_precision('high')
################################################

### Do all experiments in repeat
SAVE_PATH = "./logs"
########################################################
########################################################

#  -------------------------------------------------

def benchmark_cifar():
    global SAVE_PATH

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1, help="seed to run benchmark for all models", required=False)
    parser.add_argument("--save_dir", type=str, default="", help="directory to save the benchmarks on", required=False)

    args = parser.parse_args()

    # ------------------------------------------------------

    SEEDS = [147, 258, 369]
    # seedS = [147, 258, 369, 321, 654, 987, 741, 852, 963, 159, 357, 951, 753]

    # ------------------------------------------------------
    if args.seed >= 0:
        SEEDS = [args.seed]
    if len(args.save_dir) > 0:
        SAVE_PATH = args.save_dir
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    # ------------------------------------------------------

    EPOCHS = 2
    # EPOCHS = 200
    LR = 0.001
    num_layers = 7
    # ------------------------------------------------------
    for seed in SEEDS:
        for DS in ['c10']:
            for mlp_expansion in [1, 2]:
                num_cls = 10
                if DS == 'c100': num_cls = 100
                # ------------------------------------------------------
                seed_all(seed)
                model = MlpMixer((3, 32, 32), (4, 4), hidden_expansion=2.53, num_blocks=num_layers,
                                 num_classes=num_cls, patch_mixing="dense", channel_mixing="dense", mlp_expansion=mlp_expansion)
                model_name = f'mlp_mixer_dense_l{num_layers}_h{mlp_expansion}_{DS}_s{seed}'
                train_model(model, LR, model_name, DS, EPOCHS)

                seed_all(seed)
                model = MlpMixer((3, 32, 32), (4, 4), hidden_expansion=2.53, num_blocks=num_layers,
                                 num_classes=num_cls, patch_mixing="sparse_linear", channel_mixing="sparse_linear", mlp_expansion=mlp_expansion)
                model_name = f'mlp_mixer_sparseLinear_l{num_layers}_h{mlp_expansion}_{DS}_s{seed}'
                train_model(model, LR, model_name, DS, EPOCHS)

                seed_all(seed)
                model = MlpMixer((3, 32, 32), (4, 4), hidden_expansion=2.53, num_blocks=num_layers,
                                 num_classes=num_cls, patch_mixing="sparse_mlp", channel_mixing="sparse_mlp", mlp_expansion=mlp_expansion)
                model_name = f'mlp_mixer_sparseMlp_l{num_layers}_h{mlp_expansion}_{DS}_s{seed}'
                train_model(model, LR, model_name, DS, EPOCHS)
                # ------------------------------------------------------
            # ------------------------------------------------------
        # ------------------------------------------------------
    # ------------------------------------------------------


################################################
benchmark_cifar()
################################################
