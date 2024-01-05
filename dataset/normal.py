import torch
from tqdm import tqdm
import yaml
import numpy as np 
from dataset.dataset_rangeimages import RangeImgaes_Kitti_Dataset,worker_init_fn
if __name__ == '__main__':
    '''
    python normal.py --cfg='config/config_kitti_00.yaml'
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='../config/config_module00.yaml')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    if cfg["DATA"]["DATASET"] == 'KITTI':
        Dataset = RangeImgaes_Kitti_Dataset(cfg,"test")
    dataloader = torch.utils.data.DataLoader(Dataset, batch_size=1, shuffle=False,
         num_workers=1, pin_memory=True, sampler=None, drop_last=True,worker_init_fn=worker_init_fn)
    
    max_range = []
    sum_range = 0
    number_frames = 0
    for idx, data in enumerate(tqdm(dataloader)):
        if data[0][0].max()<250:
            max_range.append(data[0][0].max())
        sum_range = sum_range + data[0][0].sum()
        number_frames = number_frames + 1
    print("Total_Number_Range_Images:",number_frames)
    max_range = np.array(max_range)
    
    
    import matplotlib.pyplot as plt
    unique, counts = np.unique(max_range, return_counts=True)
    freq = np.asarray((unique, counts)).T
    plt.bar(unique, counts)
    plt.title('Frequency of Numbers')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.savefig('frequency_all.png')
    print("Max_Range:",max_range.max())
    print("Avg_Range:",sum_range/(number_frames*cfg["DATA"]["V_Res"]*cfg["DATA"]["H_Res"]))
    