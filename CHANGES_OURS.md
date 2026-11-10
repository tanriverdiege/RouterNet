# Changes in this fork

Fork of [YingJGuo/RouterNet](https://github.com/YingJGuo/RouterNet) (upstream commit `30f66d1`, 2025-11-11),
the official code of *RouterNet: Hierarchical Point Routing Network for Robust Vertebral Landmark
Localization on AP X-ray Images* (Guo et al., AAAI 2026). Used as a baseline in our vertebra corner
localization experiments (T1-L5, 17 vertebrae, AP and lateral X-rays). Licensed under Apache 2.0
(see `LICENSE`); all original authorship and rights remain with the original authors.

The method (network, routing, losses) is unchanged. The upstream files are untouched except one
compatibility patch; our additions are separate files.

## Modified upstream file

| File | Change | Why |
|---|---|---|
| `PreciseRoIPooling/pytorch/prroi_pool/src/prroi_pooling_gpu.c` | `THCudaCheck` -> `C10_CUDA_CHECK`, `Tensor::data<T>()` -> `data_ptr<T>()` | PyTorch >= 1.11 removed THC; the module does not compile otherwise. Computation unchanged. |

## Added files

### `utils_ours.py`, `train_ours.py`: training on our data
Copies of `utils.py` and `train.py` with:
- defaults: our data directory, batch 40 and 80 shape initializations (the README's published
  command), configurable output directory, data-loader workers and evaluation interval;
- training from ImageNet weights (`train.py` fine-tunes the authors' AASCE checkpoint);
- validation during training on `refactory/val` (`utils.py` reads `refactory/training`, so the
  best checkpoint would be chosen on training images);
- `calculate_cobb` unpacked as 7 values (it returns 7; `utils.py` expected 8 and crashed);
- `np.float` -> `float` (removed in NumPy >= 1.24);
- no hard-coded `CUDA_VISIBLE_DEVICES` (set by the cluster scheduler).

### `export_ours.py`: predictions in our evaluation format
Runs a trained model on one split and writes `predictions.jsonl` in original-image pixels. The
average training spine shape is placed either in
- `--init mean_box` (default): the average of the training images' spine boxes, identical for every
  image; uses no labels of the evaluated images. **This is the setting reported in our results.**
- `--init gt_centres`: a box from each evaluated image's own ground-truth vertebra centres, as in
  upstream `test.py`.

## Input data
Our images are ground-truth spine crops (5% padding) resized to 704 x 384, with 68 corners per image
in `<name>_corner.pts`, made from our T1-L5 dataset by `make_routernet_data.py` in the main project.
