"""
@Author Marek Szakacs
This code is part of bachelor thesis: Neural Turing Machines for modelling attention.

Run this program to generate dataset of squares and circles.

Run: python generate_imgs.py --<argument> <value> --<argument> <value> ... --<argument> <value>
Example: python generate_imgs.py --img_dim 32 --dataset_dir dataset_simple --dataset_size 30000
"""

import numpy as np
import cv2
import random
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--img_dim', type=int, default=32)
parser.add_argument('--dataset_size', type=int, default=30000)
parser.add_argument('--dataset_dir', type=str, default='dataset_simple')
args = parser.parse_args()

if os.path.exists(args.dataset_dir):
    shutil.rmtree(args.dataset_dir)
os.makedirs(args.dataset_dir)

for i in range(args.dataset_size):
    img = np.zeros((args.img_dim, args.img_dim, 1))
    img[:] = 255
    height = random.randrange(4, args.img_dim/2)
    width = random.randrange(4, args.img_dim/2)
    start_pos = (random.randrange(0, args.img_dim - width), random.randrange(0, args.img_dim - height))
    end_pos = (start_pos[0] + width, start_pos[1] + height)
    cv2.rectangle(img, start_pos, end_pos, (0, 0, 0), 1)
    cv2.imwrite(args.dataset_dir + "/rectangle" + str(i) + ".png", img)

for i in range(args.dataset_size):
    img = np.zeros((args.img_dim, args.img_dim, 1))
    img[:] = 255
    radius = random.randrange(4, args.img_dim / 2 - 4)
    cv2.circle(img, (random.randrange(radius, args.img_dim - radius), random.randrange(radius, args.img_dim-radius)),
               radius, (0, 0, 0), 1)
    cv2.imwrite(args.dataset_dir + "/circle" + str(i) + ".png", img)

