### ws_proc

Generate dataset, follow README from ws_proc.

```bash
python build_training_set.py
python set_attr_table.py
python set_pga.py
python subset_data_vs30.py
```

### eval_cond_ffn1d.py

Use as base file (GANO). Should work with the Japanese data.

Parameters alpha and tau might need some adaption in order to work well (params for GRF).

Neural operator (generator): change modes and wd

### gan_cond_fno1d.py

Basic GAN architecture

### wgan_cond_1d_eval.py

GAN version take from `wgan_cond_1d_eval.py` (1 component version).
When preparing dataset, downsample only 4x (25 Hz instead of 20 Hz). Distance only (input parameters), no magnitude yet.

### General
No magnitude in GANO so far.

### Next steps

- Get code to run on Sisma:
    - Prepare data
    - Get GAN to run