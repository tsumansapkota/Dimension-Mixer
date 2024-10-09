## Patch Only MLP-Mixer

This sub-directory includes code and experiments for Ordinary MLP-Mixer Model and Patch-Only MLP-Mixer.

### Benchmark

From the directory `PatchOnlyMlpMixer` run:   

```python benchmark_mixers.py --seed [SEED] --save_dir [SAVE DIRECTORY]```

By default, the `SEED` is `None`, which used `[147, 258, 369]`, and `SAVE DIRECTORY` is `./logs` 

For Reproducibility Experiments use `SEED` from the list: `[147, 258, 369]` in each run.