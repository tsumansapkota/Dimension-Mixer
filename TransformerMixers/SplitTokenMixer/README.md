## Token Parallel Attention

This sub-directory includes code and experiments for Token Parallel Attention.

### Benchmark

From the directory `TransformerMixers/SplitTokenMixer/` run:   

```python benchmark_attention_token.py --seed [SEED] --save_dir [SAVE DIRECTORY]```

By default, the `SEED` is `None` which uses `[147, 258, 369]`, and `SAVE DIRECTORY` is `./logs` 

For Reproducibility Experiments use `SEED` from the list: `[147, 258, 369]` in each run.