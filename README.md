# NeRI

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Official Implementation of "NERI: IMPLICIT NEURAL REPRESENTATION OF LIDAR POINT CLOUD USING RANGE IMAGE SEQUENCE"

![Local Image](/fig/Framework.png)

## Get Started
### Data Preparation
The dataset used in this work can be downloaded from [KITTI](https://www.cvlibs.net/datasets/kitti/index.php).
It should be organized as follows:
- `sequence/00/`
  - `velodyne/`
    - `000000.bin`
    - `000001.bin`
    - `000002.bin`
    - `......`
  - `calib.txt`
  - `pose.txt`
- `......`
### Dependencies
You can install the dependencies using the following command：
```bash
conda create -n neri
conda activate neri
pip install -r requirements.txt 
```

### Training
Here is an example command for training:
```bash
python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --temporal_embed='1.25_20' --translation_embed='1.25_30' --rotation_embed='1.25_30'  --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2
```
### Testing
Here is an example command for testing:
```bash
python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --temporal_embed='1.25_20' --translation_embed='1.25_30' --rotation_embed='1.25_30'  --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2 --eval_only --quant_bit=-1
```
```bash
python train.py -e 600 --outf tp --stem_dim_num 64_1 --fc_hw_dim 4_125_26  --single_res --act swish --eval_freq=1 --temporal_embed='1.25_20' --translation_embed='1.25_30' --rotation_embed='1.25_30'   --segmentation --cfg='config/kitti_00.yaml' --strides 2 2 2 2 --eval_only --quant_mode='pw-1' --quant_bit=16
```
The bpp (bits per point) can be calculated by:

```math
  Bpp = ModelSize \times QuantizationPrecision \times EntropyEfficiency / TotalNumPoints
```

## Contact
If you have any questions, concerns, or would like to contribute to this project, feel free to reach out. You can contact us at：
- **Email:** chentong@nju.edu.cn

