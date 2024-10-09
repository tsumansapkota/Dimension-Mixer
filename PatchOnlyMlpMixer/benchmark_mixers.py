import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from mlp_mixers import MlpMixer, PatchMlpMixer

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
#  -------------------------------------------------
def benchmark_cifar():
    global SAVE_PATH

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default=-1, help = "seed to run benchmark for all models", required = False)
    parser.add_argument("--save_dir", type = str, default="", help = "directory to save the benchmarks on", required = False)

    args = parser.parse_args()

    #------------------------------------------------------

    SEEDS = [147, 258, 369]
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
    EPOCHS = 200
    LR = 0.001

    #------------------------------------------------------
    for seed in SEEDS:
        for DS in ['c10']:
        # for DS in ['c10','c100']:
            for num_layers in [7, 10]:
                num_cls = 10
                if DS=='c100': num_cls = 100
                #------------------------------------------------------
                
                ## Original Mixer with 4*9=36 image resize --> with similar number of parameters (balanced with PatchOnly).
                seed_all(seed)
                model = MlpMixer((3, 4*9, 4*9), (4, 4), hidden_expansion=3.0, num_blocks=num_layers, num_classes=num_cls)
                model_name = f'original_mixer0_l{num_layers}_{DS}_s{seed}'
                train_model(model, LR, model_name, DS, EPOCHS)

                ## Original Mixer with 4*8=32 image resize
                seed_all(seed)
                model = MlpMixer((3, 32, 32), (4, 4), hidden_expansion=3.2, num_blocks=num_layers, num_classes=num_cls)
                model_name = f'original_mixer1_l{num_layers}_{DS}_s{seed}'
                train_model(model, LR, model_name, DS, EPOCHS)

                ## Patch Only Mixer with 5*7=35 image resize
                seed_all(seed)
                model = PatchMlpMixer((3, 35, 35), patch_sizes=[5,7], hidden_channels=3, num_blocks=num_layers, num_classes=num_cls)
                model_name = f'patchonly_mixer0_l{num_layers}_{DS}_s{seed}'
                train_model(model, LR, model_name, DS, EPOCHS)

                #------------------------------------------------------
            #------------------------------------------------------
        #------------------------------------------------------
    #------------------------------------------------------

################################################

benchmark_cifar()

################################################
################################################
################################################
