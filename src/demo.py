#! /user/bin/env python
# encoding=utf-8

# ------------------------------------
# caffe squeezeDet
# Written by XuSenhai
# ------------------------------------

import _init_paths
import caffe
import numpy as np
import os
import cv2
import argparse
from config.config import cfg
from utils.timer import Timer
from test import im_detect
import pdb

CLASSES = ('__background__', \
           'car', 'pedestrain', 'cyclist')

NETS = {'squeeze': ('squeezedet.prototxt', \
                    'squeezeDet.caffemodel')}

def parse_args():
    parser = argparse.ArgumentParser(description = 'Squeeze_Det')

    parser.add_argument('--gpu', dest = 'gpu_id', default = 1, type = int)
    parser.add_argument('--net', dest = 'demo_net', default = 'squeeze', type = str)
    
    args = parser.parse_args()

    return args

def demo(net, image_name):
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)

    im = cv2.imread(im_file)
    
    det_boxes, det_probs, det_class = im_detect(net, im)
    
    print('det_boxes: ', det_boxes)
    print('det_probs: ', det_probs)
    print('det_class: ', det_class)

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'prototxt', \
                            NETS[args.demo_net][0])

    caffemodel = os.path.join(cfg.ROOT_DIR, 'model_checkpoints', \
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found'.format(caffemodel)))

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    
    #caffe.set_mode_cpu()

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print('\n Loaded network {:s}'.format(caffemodel))

    im_names = ['sample.png']

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(net, im_name)
    

