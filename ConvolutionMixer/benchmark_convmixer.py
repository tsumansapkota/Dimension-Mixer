import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import resnet_conv_mix
import resnet_block_mix

sys.path.append("../")
import utils

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

    EPOCHS = 200
    LR = 0.1 ## for SGD

    #------------------------------------------------------
    for seed in SEEDS:
        for DS in ['c10']:
            num_cls = 10
            #------------------------------------------------------

            for i in range (10):
                utils.seed_all(seed)
                ## Default -> planes=16, G=[4, 8, 8]
                if i == 0:
                    model = resnet_conv_mix.CifarResNet(resnet_conv_mix.BasicBlock, [3, 3, 3], num_classes=num_cls, mixer=False) ## default cifar
                    model_name = f'resnet20_p16_{DS}_s{seed}'
                elif i == 1:
                    model = resnet_conv_mix.CifarResNet(resnet_conv_mix.BasicBlock, [3, 3, 3], num_classes=num_cls, planes=32, mixer=True, G=[8, 8, 16])
                    model_name = f'resnet20_p32_convmix_{DS}_s{seed}'
                elif i == 2:
                    model = resnet_block_mix.CifarResNet(resnet_block_mix.BasicBlock, [3, 3, 3], num_classes=num_cls, planes=32, group_sizes=[4, 4, 8])
                    model_name = f'resnet20_p32_blockmix_{DS}_s{seed}'
                elif i == 3:
                    model = resnet_block_mix.CifarResNet(resnet_block_mix.BasicBlock, [2, 2, 2], num_classes=num_cls, planes=32, group_sizes=[8, 8, 16])
                    model_name = f'resnet17_p32_blockmix_{DS}_s{seed}'
                elif i == 4:
                    model = resnet_conv_mix.CifarResNet(resnet_conv_mix.BasicBlock, [3, 3, 3], num_classes=num_cls, planes=32, mixer=False)
                    model_name = f'resnet20_p32_{DS}_s{seed}'
                elif i == 5:
                    model = resnet_block_mix.CifarResNet(resnet_block_mix.BasicBlock, [3, 3, 3], num_classes=num_cls, planes=64, group_sizes=[4, 8, 8])
                    model_name = f'resnet20_p64_blockmix_{DS}_s{seed}'
                elif i == 6:
                    model = resnet_conv_mix.CifarResNet(resnet_conv_mix.BasicBlock, [3, 3, 3], num_classes=num_cls, planes=64, mixer=True, G=[8, 16, 16])
                    model_name = f'resnet20_p64_convmix_{DS}_s{seed}'
                elif i == 7:
                    model = resnet_conv_mix.CifarResNet(resnet_conv_mix.BasicBlock, [4, 4, 4], planes=32, num_classes=num_cls, mixer=False)
                    model_name = f'resnet23_p32_{DS}_s{seed}'
                elif i == 8:
                    model = resnet_conv_mix.CifarResNet(resnet_conv_mix.BasicBlock, [4, 4, 4], planes=64, num_classes=num_cls, mixer=True, G=[8, 16, 16])
                    model_name = f'resnet23_p64_convmix_{DS}_s{seed}'
                elif i == 9:
                    model = resnet_block_mix.CifarResNet(resnet_block_mix.BasicBlock, [4, 4, 4], planes=64, num_classes=num_cls, group_sizes=[8, 16, 16])
                    model_name = f'resnet23_p64_blockmix_{DS}_s{seed}'
                else:
                    continue
                print(i)
                utils.train_model(model, LR, f"{i}_"+model_name, DS, EPOCHS, optim="SGD")
                #------------------------------------------------------
            #------------------------------------------------------
        #------------------------------------------------------
    #------------------------------------------------------

################################################

benchmark_cifar()

################################################


