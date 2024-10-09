#### Transformers/Attention
cd TransformerMixers
cd ButterflyAttention
python benchmark_attention_butterfly.py
python benchmark_attention_memory.py
cd ../SplitTokenMixer
python benchmark_attention_token.py
cd ../../

##### Non-Linear in MLP mixers
cd SparseNonLinearMixer
python benchmark_sparse_mlp_mixers.py
cd ../

##### Patch Only MLP mixer
cd PatchOnlyMlpMixer
python benchmark_mixers.py
cd ../

##### Convolution Mixer
cd ConvolutionMixer
python benchmark_convmixer.py
cd ../

##### Sparse Linear mixer
cd SparseLinearMixer
python benchmark_linear_mixer.py
cd ../
##### END
