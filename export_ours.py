"""Export RouterNet predictions for one split as predictions.jsonl, in original-image pixels.

Runs the trained model on our RouterNet data (704 x 384 crops made by make_routernet_data.py) and maps
the 68 predicted corners back to the original X-ray with transforms.csv:
    original = resized / scale + (x0, y0).

The initial shape (the average training spine) must be placed in a box before the network refines it:
    --init mean_box    the average of the training images' spine boxes, the same box for every image;
                       uses no labels of the evaluated images (the setting reported in our paper)
    --init gt_centres  a box built from each evaluated image's own ground-truth vertebra centres,
                       as RouterNet's test.py does (uses the evaluated image's labels)

Run from this folder in the RouterNet environment, on a GPU:
    python export_ours.py --split test --init mean_box --out <dir>
"""
import argparse
import csv
import json
import os
import sys

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--split", choices=("train", "val", "test"), required=True)
p.add_argument("--out", required=True, help="output directory for predictions.jsonl")
p.add_argument("--init", choices=("mean_box", "gt_centres"), default="mean_box")
p.add_argument("--data", default="/ceph/nacl/users/etanriverdi/routernet_data/", help="with trailing /")
p.add_argument("--ckpt", default="/ceph/nacl/users/etanriverdi/model_checkpoints/routernet/model/Oct23-1627_root_8/best_smape")
args = p.parse_args()
sys.argv = [sys.argv[0]]  # utils.py parses the command line at import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # RouterNet modules import each other by name

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from utils import *  # noqa: E402,F401,F403  (LoadModels, TestGlobalRegression, GenerateBBox, ...)


def spine_box(f):
    """GenerateBBox of one image's ground-truth vertebra centres (RouterNet's placement box)."""
    corners = ReadCornerShape(args.data + f.split(".jpg")[0] + "_corner.pts", offsets=[0, 0])
    return GenerateBBox(GetCenterFromCorner(corners))


T = {r["file"]: r for r in csv.DictReader(open(args.data + "transforms.csv"))}
files = [l.strip() for l in open(args.data + f"{args.split}_img_list") if l.strip()]
imgs = []
for f in files:
    im = cv2.imread(args.data + f, 0)
    assert im.shape == (704, 384), f
    imgs.append(im.astype(np.float64))

if args.init == "gt_centres":
    bboxes = [spine_box(f) for f in files]
else:
    train = [l.strip() for l in open(args.data + "train_img_list") if l.strip()]
    mean_box = np.mean([spine_box(f) for f in train], axis=0)
    print("training boxes:", len(train), "mean box (x1, y1, x2, y2):", mean_box.round(1))
    bboxes = [mean_box for _ in files]

nets_, mrc, mrk = LoadModels(args.ckpt)
mean_c = np.array([Shape2Absolute(mrc, b) for b in bboxes])
mean_k = np.array([GenerateCornerVec(Shape2Absolute(mrk, b))[0] for b in bboxes])
_, corners = TestGlobalRegression(imgs, mean_c, mean_k, nets_)

NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "predictions.jsonl"), "w") as fo:
    for f, pc in zip(files, corners):
        t = T[f]
        x0, y0, sx, sy = float(t["x0"]), float(t["y0"]), float(t["scale_x"]), float(t["scale_y"])
        kps = []
        for k in range(17):
            d = {"label": str(k + 1), "score": 1.0}
            for j, n in enumerate(NAMES):
                x, y = pc[4 * k + j]
                d[n] = {"x": float(x / sx + x0), "y": float(y / sy + y0), "label": n}
            kps.append(d)
        fo.write(json.dumps({"image_path": t["orig_jpeg"], "spinetk_id": t["spinetk_id"], "view": t["view"],
                             "image_size": [int(t["orig_width"]), int(t["orig_height"])], "crop_box": None,
                             "checkpoint": args.ckpt, "init": args.init, "keypoints": kps}) + "\n")
print("wrote", len(files), "to", args.out)
