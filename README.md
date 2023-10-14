# LightFC

The official implementation of LightFC

## News

- 14 Oct 2023:  our code is available now
- 09  Oct 2023:  our manuscript have submitted to [arxiv](https://arxiv.org/abs/2310.05392)

## Install the environment

**Option1**: Use the Anaconda
```
conda create -n lightfc python=3.9
conda activate lightfc
bash install.sh
```

## Data Preparation
   ```
   ${YOUR_DATASETS_ROOT}
        -- lasot
            |-- airplane
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            ...
            |-- TRAIN_11
            |-- TEST
        -- ...
   ```
## Set project paths

Go to these two files, and modify the paths
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train LightFC
Training with multiple GPUs using DDP

```
python tracking/train.py --script LightFC --config mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou --save_dir . --mode multiple --nproc_per_node 2 
```

## Test and evaluate LightFC on benchmarks
Go to **tracking/test.py** and modify the parameters
```
python tracking/test.py
```

Then Go to **tracking/analysis_results.py** and modify the parameters
```
python tracking/analysis_results.py
```
## Test FLOPs, Params, and Speed
```
# Params and FLOPs
python tracking/profile_model.py
# Speed
python tracking/speed.py
```

## Acknowledgments
* Thanks for the great [stark](https://github.com/researchmm/Stark) and [ostrack](https://github.com/botaoye/OSTrack) Libraries, which helps us to quickly implement our ideas.
