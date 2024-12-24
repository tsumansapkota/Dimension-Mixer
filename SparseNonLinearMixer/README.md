## Sparse Non-Linear in MLP-Mixer

This sub-directory includes code and experiments for Ordinary MLP-Mixer Model and Patch-Only MLP-Mixer.

### Benchmark

From the directory `SparseNonLinearMixer` run:   

```python benchmark_sparse_mlp_mixers.py --seed [SEED] --save_dir [SAVE DIRECTORY]```

By default, the `SEED` is `None`, which uses all required seeds, and `SAVE DIRECTORY` is `./logs` 

For Reproducibility Experiments use `SEED` from the list: `[147, 258, 369, 321, 654, 987, 741, 852, 963, 159, 357, 951, 753]` in each run.