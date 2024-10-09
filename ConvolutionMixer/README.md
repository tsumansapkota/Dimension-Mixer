## Butterfly Convolution Mixer - CIFAR

This sub-directory includes code and experiments for Butterfly Convolution.

### Benchmark

From the directory `ConvolutionMixer/` run:   

```python benchmark_convmixer.py --seed [SEED] --save_dir [SAVE DIRECTORY]```

By default, the `SEED` is `None`, which uses `[147, 258, 369]`, and `SAVE DIRECTORY` is `./logs` 

For Reproducibility Experiments use `SEED` from the list: `[147, 258, 369]` in each run.