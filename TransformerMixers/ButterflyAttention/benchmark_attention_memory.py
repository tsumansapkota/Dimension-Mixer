import os, time

SEEDS = [147]
### CONFIGS TUPLE
## (DS, num_layers, patch_size)
configs = [
    ('c10', 8, 4),
    ('c10', 4, 4),
    ('c10', 4, 2),
    ('c10', 4, 1),
    ('c100', 4, 4),
    ('c100', 4, 2),
]
patch_expansion_dict = {16: 1024, 8: 256, 4: 128, 2: 64, 1: 64}
imsize = (3, 32, 32)

for config in configs:
    DS, num_layers, patch_size = config
    # ------------------------------------------------------
    for butterfly_att in [False, True]:
        cmd = f"python single_attention_mem.py --dataset '{DS}' --patch_size {patch_size} --num_layers {num_layers} --seed 147"
        if butterfly_att:
            cmd += " --sparse_att"
        os.system(cmd)
        # time.sleep(10) ## to let gpu cool a bit
