#!/usr/bin/env python
# encoding=utf-8

# ---------------------------------------
# caffe squeezeDet
# Written by XuSenhai
# ---------------------------------------

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))

__C.DATA_DIR = osp.join(__C.ROOT_DIR, 'data')

# Pixel mean values for BGR
__C.PIXEL_MEANS = np.array([[[103.939, 116.779, 123.68]]])

# Image width
__C.IMAGE_WIDTH = 1248

# Image height
__C.IMAGE_HEIGHT = 384

# Classes number
__C.NUM_CLASSES = 3

# Anchors number
__C.NUM_ANCHORS = 16848

# Top detected objects number
__C.TOP_N_DETECTION = 64

# Detection probability threshold
__C.PROB_THRESHOLD = 0.005

# Detection NMS threeshold
__C.NMS_THRESHOLD = 0.4

# Draw bounding boxes probability threshold
__C.DRAW_THRESHOLD = 0.4
