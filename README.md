# LightFC

The official implementation of LightFC

## News

- 14 Oct 2023:  our code is available now
- 09 Oct 2023:  our manuscript have submitted to [arxiv](https://arxiv.org/abs/2310.05392)
- 12 Jan 2024:  lightfc-vit with higher performance is released !
## Install the environment

**Option1**: Use the Anaconda
```
conda create -n lightfc python=3.9
conda activate lightfc
bash install.sh
```

## Data Preparation
   Follow [stark](https://github.com/researchmm/Stark) and [ostrack](https://github.com/botaoye/OSTrack) frameworks to set your datasets

## File directory

Project file directory should be like

   ```
   ${YOUR_PROJECT_ROOT}
        -- experiments
            |-- lightfc
        -- external
            |-- vot20st
        -- lib
            |--models
            ...
        -- outputs (download and unzip the output.zip to obtain our checkpoints and row results)
            |--checkpoints
                |--...
            |--test
                |--...
        -- pretrained_models (if you want to train lightfc, put pretrained model here)
            |--mobilenetv2.pth (from torchvision model)
            ...    
        -- tracking
            ...
   ```

Download lightfc checkpoint and raw results at [Google Drive](https://drive.google.com/file/d/1ns7NQJCt078547X483skqjX1qM1rBqLP/view)

Download lightfc-vit checkpoint and raw results at [Google Drive](https://drive.google.com/file/d/1J4ubpqN4yKjETiHkVEsP2M_9xHEfS8nR/view?usp=sharing)

Then go to these two files, and modify the paths
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```


## Train LightFC
Training with multiple GPUs using DDP
```
python tracking/train.py --script LightFC --config mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou --save_dir . --mode multiple --nproc_per_node 2 
```
If you want to train lightfc, please download https://download.pytorch.org/models/mobilenet_v2-b0353104.pth rather than https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth


## Test and evaluate LightFC on benchmarks
Go to **tracking/test.py** and modify the parameters
```
python tracking/test.py
```

Then go to **tracking/analysis_results.py** and modify the parameters
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
