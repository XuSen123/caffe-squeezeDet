#!/usr/bin/env python
# encoding=utf-8

# -----------------------------
# caffe squeezedet
# Written bu XuSenhai
# -----------------------------

import caffe
from config.config import cfg
from utils.utils import softmax, sigmoid, set_anchors, nms
from utils.utils import bbox_transform, bbox_transform_inv, clip_box
from utils.timer import Timer
import os
import cv2
import numpy as np
import pdb

def filter_prediction(boxes, probs, cls_idx):
    """
    Filter bounding boxes with probability threshold and nms
    Args:
        boxes: [BATCH, 4], (cx, cy, w, h)
        probs: [BATCH, CLASS_NUM], class probability
        cls_idx: array of class indices
    Return:
        final_boxes: filtered bounding boxes
        final_probs: filtered probabilities
        final_cls_idx: filtered class indices
    """ 
    if cfg.TOP_N_DETECTION < len(probs) and cfg.TOP_N_DETECTION > 0:
        order = probs.argsort()[:-cfg.TOP_N_DETECTION-1:-1]
        probs = probs[order]
        boxes = boxes[order]
        cls_idx = cls_idx[order]
    else:
        filtered_idx = np.nonzero(probs > cfg.PROB_THRESHOLD)[0]
        probs = probs[filtered_idx]
        boxes = boxes[filtered_idx]
        cls_idx = cls_idx[filtered_idx]

    final_boxes = []
    final_probs = []
    final_cls_idx = []

    for c in range(cfg.NUM_CLASSES):
        idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
        keep = nms(boxes[idx_per_class], probs[idx_per_class], cfg.NMS_THRESHOLD)
        for i in range(len(keep)):
            if keep[i]:
                final_boxes.append(boxes[idx_per_class[i]])
                final_probs.append(probs[idx_per_class[i]])
                final_cls_idx.append(c)
    return final_boxes, final_probs, final_cls_idx


def im_detect(net, im):
    im = im.astype(np.float32, copy = False)
    im = cv2.resize(im, (int(cfg.IMAGE_WIDTH), int(cfg.IMAGE_HEIGHT)))
    im = im - cfg.PIXEL_MEANS
    im_ = np.transpose(im, (2, 0, 1))
    im_ = im_[np.newaxis, ...]
    
    conv1_shadow = np.zeros((1, 64, 192, 624), dtype = np.float32)
    
    net_time = Timer()
    npy_time = Timer()

    # reshape network inputs
    net_time.tic()
    net.blobs['data'].reshape(*(im_.shape))
    net.blobs['conv1_shadow'].reshape(*(conv1_shadow.shape))
    
    # do forward
    forward_kwargs = {'data': im_.astype(np.float32, copy = False), \
                      'conv1_shadow': conv1_shadow}

    blobs_outs =net.forward(**forward_kwargs)
    net_time.toc()
    
    npy_time.tic()
    pred_class_probs = blobs_outs['pred_class_probs']
    pred_conf = blobs_outs['pred_conf']
    pred_box_delta = blobs_outs['pred_box_delta']
    
    # For pred class probs part
    pred_class_probs = np.transpose(pred_class_probs, (0, 2, 3, 1))
    pred_class_probs = np.reshape(pred_class_probs, (cfg.NUM_ANCHORS, 3))
    pred_class_sft_probs = softmax(pred_class_probs) 
    pred_class_sft_probs = np.reshape(pred_class_sft_probs, (-1, cfg.NUM_ANCHORS, cfg.NUM_CLASSES))
    
    # For pred conf part
    pred_conf = np.transpose(pred_conf, (0, 2, 3, 1))
    pred_conf = np.reshape(pred_conf, (-1, cfg.NUM_ANCHORS))
    pred_sgm_conf = sigmoid(pred_conf)
    
    # For pred box delta
    pred_box_delta = np.transpose(pred_box_delta, (0, 2, 3, 1))
    pred_box_delta = np.reshape(pred_box_delta, (-1, cfg.NUM_ANCHORS, 4))
    delta_x = pred_box_delta[:, :, 0]
    delta_y = pred_box_delta[:, :, 1]
    delta_w = pred_box_delta[:, :, 2]
    delta_h = pred_box_delta[:, :, 3]
   
    # Get anchors
    ANCHORS = set_anchors()
    anchor_x = ANCHORS[:, 0]
    anchor_y = ANCHORS[:, 1]
    anchor_w = ANCHORS[:, 2]
    anchor_h = ANCHORS[:, 3]
    box_center_x = anchor_x + delta_x * anchor_w
    box_center_y = anchor_y + delta_y * anchor_h
    box_width = anchor_w * np.exp(delta_w)
    box_height = anchor_h * np.exp(delta_h)
    
    # convert (centerx, centery, w, h) into (xmin, ymin, xmax, ymax)
    xmins, ymins, xmaxs, ymaxs = bbox_transform([box_center_x, box_center_y, \
                                                 box_width, box_height])
    # clip bounding boxes
    xmins, ymins, xmaxs, ymaxs = clip_box([xmins, ymins, xmaxs, ymaxs])
    
    # convert (xmin, ymin, xmax, ymax) into (centerx, centery, w, h)
    bbox = bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])
    det_bbox = np.array(bbox)
    det_bbox = np.transpose(det_bbox, (1, 2, 0))
    
    # get probs
    det_probs = np.multiply(pred_class_sft_probs, \
                np.reshape(pred_sgm_conf, (-1, cfg.NUM_ANCHORS, 1)))
    det_probs_max = np.max(np.reshape(det_probs, (-1, 3)), axis = 1)
    det_probs_max = np.reshape(det_probs_max, (1, -1))

    # get class
    det_class = np.argmax(det_probs, 2)
    
    # filter bounding boxes with nms
    final_boxes, final_probs, final_class = filter_prediction(det_bbox[0], \
                                                    det_probs_max[0], det_class[0])
    
    # filter bounding boxes with probability threshold
    keep_idx = [idx for idx in range(len(final_probs))\
                    if final_probs[idx] > cfg.DRAW_THRESHOLD]
    final_boxes = [final_boxes[idx] for idx in keep_idx]
    final_probs = [final_probs[idx] for idx in keep_idx]
    final_class = [final_class[idx] for idx in keep_idx]
    
    npy_time.toc()
    print('net time: {:.3f}, npy time: {:.3f}'.format(net_time.average_time, npy_time.average_time))

    return final_boxes, final_probs, final_class 
