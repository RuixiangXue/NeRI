# NeRI

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Official Implementation of "NERI: IMPLICIT NEURAL REPRESENTATION OF LIDAR POINT CLOUD USING RANGE IMAGE SEQUENCE"

## Data Preparation
The dataset used in this work can be downloaded from [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)
- `data/raw/`: 存储原始数据的目录。
  - `images/`: 存储原始图像文件。
  - `annotations/`: 存储与图像相关的标注文件，如 XML 或 JSON

## Get Started

### Dependencies

```bash
conda create -n neri
conda activate neri
pip install -r requirements.txt 
```

### Training
```bash
python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --temporal_embed='1.25_20' --translation_embed='1.25_30' --rotation_embed='1.25_30'  --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2
```
### Testing

```bash
python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --temporal_embed='1.25_20' --translation_embed='1.25_30' --rotation_embed='1.25_30'  --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2 --eval_only --quant_bit=-1
```
```bash
python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --temporal_embed='1.25_20' --translation_embed='1.25_30' --rotation_embed='1.25_30'   --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2 --eval_only --quant_mode='pw-1' --quant_bit=16
```
