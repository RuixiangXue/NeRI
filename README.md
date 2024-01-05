python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --embed='1.25_80'  --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2

python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --embed='1.25_80'  --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2 --eval_only --quant_bit=-1

python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --temporal_embed='1.25_20' --translation_embed='1.25_30' --rotation_embed='1.25_30'   --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2 --eval_only --quant_mode='pw-1' --quant_bit=16

# NeRI
Official Implementation of "NERI: IMPLICIT NEURAL REPRESENTATION OF LIDAR POINT CLOUD USING RANGE IMAGE SEQUENCE"

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

简短的项目描述一句话。

## 特色

- 列举项目的主要特色和亮点。

## 快速开始

这里提供如何快速在本地安装和运行项目的简要说明。

```bash
git clone https://github.com/你的用户名/你的项目.git
cd 你的项目
npm install  # 或者使用 yarn
npm start    # 或者使用 yarn start
