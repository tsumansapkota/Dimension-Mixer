import subprocess as sp
import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import numpy as np
import time, json

sys.path.append("../../")
from utils import seed_all
import transformers_lib_butterfly as transformers

torch.use_deterministic_algorithms(True) ## remove this for compile
torch.set_float32_matmul_precision('high')

def get_memory_used(cuda=0):
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values[cuda]


def benchmark_memory(dataset: str, patch_size: int, num_layers: int, SEED: int, sparse_att: bool = False, pos_emb: bool = False, cuda: int = 0):
    device = torch.device(f"cuda:{cuda}")

    if sparse_att:
        assert num_layers % 2 == 0, 'number of blocks on sparse transformer is (x2)/2 hence it must be even'
        num_layers_ = num_layers // 2
    else:
        num_layers_ = num_layers

    BS = None
    NC = None
    imsize = (3, 32, 32)
    expansion_dict = {16: 1024, 8: 256, 4: 128, 2: 64, 1: 64}
    expansion = expansion_dict[patch_size]
    seed_all(SEED)

    if dataset == 'c10':
        NC = 10
        BS = 64
    elif dataset == 'c100':
        NC = 100
        BS = 128

    ### Now create models
    seq_len = (imsize[-1] * imsize[-2]) // (patch_size * patch_size)
    mlp_dim = expansion
    print(seq_len, mlp_dim)

    if sparse_att:
        seq_len = int(2 ** np.ceil(np.log2(np.sqrt(seq_len))))
    # if sparse_mlp:
    #     mlp_dim = int(2 ** np.ceil(np.log2(np.sqrt(expansion))))


    # _x = torch.randn(BS, *imsize)  # .to(device)
    # _y = torch.randint(NC, (BS,))
    # inputs, targets = _x.to(device), _y.to(device)
    #     print("Output: ",vit_mixer(_x).shape)

    mem_begin = get_memory_used(cuda=cuda)
    ### starting with createion of model
    torch.manual_seed(SEED)
    model = transformers.Mixer_ViT_Classifier(imsize,
                                 patch_size=[patch_size] * 2,
                                 hidden_channel=expansion,
                                 num_blocks=num_layers_,
                                 num_classes=NC,
                                 block_seq_size=seq_len,
                                 block_mlp_size=mlp_dim,
                                 pos_emb=pos_emb).to(device)
    # model = torch.compile(model)
    model.train()
    # _ = model(inputs) ### after torch compile
    # torch.cuda.empty_cache()
    # time.sleep(2) ## to cleanup the compile mem
    # print("Train model compile")

    num_params = sum(p.numel() for p in model.parameters())
    print("number of params: ", num_params)

    _a = 'att'
    if sparse_att: _a = 'sAtt'
    model_name = f"mem_ViT_nPE_{dataset}_patch{patch_size}_l{num_layers}_{_a}_s{SEED}"
    print(f"Model Name: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training
    ### test time taken for multiple iterations
    time_taken = []
    for i in range(50):
        inputs = torch.randn(BS, *imsize).to(device)
        targets = torch.randint(NC, (BS,)).to(device)
        torch.cuda.synchronize()
        start = time.time()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        start = time.time() - start
        time_taken.append(start)
    train_time = np.min(time_taken)

    mem_end = get_memory_used(cuda=cuda)
    print(f"mem begin: {mem_begin}  end: {mem_end}")

    model.eval()
    # _ = model(inputs) ### after torch compile
    # torch.cuda.empty_cache()
    # time.sleep(2)
    # print("Eval model compile")

    time_taken = []
    for i in range(50):
        inputs = torch.randn(BS, *imsize).to(device)
        # targets = torch.randint(NC, (BS,)).to(device)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()

            _ = model(inputs)

            torch.cuda.synchronize()
            start = time.time() - start
            time_taken.append(start)

    test_time = np.min(time_taken)

    # filename = f"./logs_benchmark_memory_compile.json"
    filename = f"./logs_benchmark_memory_eager.json" ## if running without torch.compile()
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump({}, f, indent=0)

    with open(filename, 'r+') as f:
        file_data = json.load(f)
        file_data[f"{model_name}"] = {'memory': mem_end - mem_begin, 'time_train': train_time, 'time_test': test_time}
        f.seek(0)
        json.dump(file_data, f, indent=0)

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, help = "dataset -> c10 or c100", required = True)
    parser.add_argument("--patch_size", type = int, help = "size of patch -> e.g. 4, 2, 1", required = True)
    parser.add_argument("--num_layers", type = int, help = "layers of transformer -> e.g. 4, 8", required = True)
    parser.add_argument("--seed", type = int, help = "seed to run benchmark for all models", required = True)
    parser.add_argument("--sparse_att", help = "use for running butterfly attention", default=False, action='store_true')
    args = parser.parse_args()

    benchmark_memory(args.dataset, args.patch_size, args.num_layers, args.seed, args.sparse_att, pos_emb=False, cuda=0) ## change your device here
