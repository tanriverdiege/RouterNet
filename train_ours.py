"""Train RouterNet on our data (routernet_data: T1-L5, AP and lateral together).

Copy of train.py using utils_ours.py, and starting from ImageNet weights (train.py
fine-tunes their AASCE model, ./model/Jul25-1445_root_8/best_corner). Settings are in
utils_ours.py; the README's published command is the default (batch 40, 80 initializations).

    python train_ours.py                       # full run
    python train_ours.py --param_epoch 1 -aug_num 1 --out_dir <dir>   # quick smoke test
"""
from utils_ours import *

if __name__ == '__main__':
    imgs, bboxes, mean_rshape_center, mean_rshape_corner, center_shapes, corner_shapes = LoadData(data_path=args.data_dir)
    train_imgs, train_bboxes, train_init_center, train_init_corner, gt_center_shapes, gt_corner_shapes = GetTrainData(
        imgs, bboxes, center_shapes, corner_shapes, mean_rshape_center, mean_rshape_corner)
    net = GlobalRegression(train_imgs, train_init_center, train_init_corner,
                           gt_center_shapes, gt_corner_shapes, mean_rshape_center, mean_rshape_corner, pre_train=None)
