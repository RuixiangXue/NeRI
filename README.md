# NeRI

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Official Implementation of "NERI: IMPLICIT NEURAL REPRESENTATION OF LIDAR POINT CLOUD USING RANGE IMAGE SEQUENCE"

## Data Preparation



## Get Started

### Dependencies

```bash
conda create -n neri
conda activate neri
pip install -r requirements.txt 
```

### Training

### Testing

```bash
python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --embed='1.25_80'  --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2

python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --embed='1.25_80'  --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2 --eval_only --quant_bit=-1

python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --temporal_embed='1.25_20' --translation_embed='1.25_30' --rotation_embed='1.25_30'   --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2 --eval_only --quant_mode='pw-1' --quant_bit=16
```
