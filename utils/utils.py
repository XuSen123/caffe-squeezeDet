#encdoing=utf-8
# -----------------------------
# caffe squeezeDet
# Written by XuSenhai
# -----------------------------

import numpy as np
import pdb
from config.config import cfg

def softmax(data):
    """
    Softmax function declear here.
    Args:
        data: input blobs with B x C
    """
    exp_data = np.exp(data)
    exp_sum = np.sum(exp_data, axis = 1)
    exp_sum = exp_sum[:, np.newaxis]

    data_softmax = exp_data / exp_sum
      
    return data_softmax

def sigmoid(data):
    exp_data = 1 + np.exp(-1. * data)
    data_sigmoid = 1.0 / exp_data

    return data_sigmoid

def set_anchors():
    H, W, B = 24, 78, 9
    anchor_shapes = np.reshape(\
            [np.array(\
                [[  36.0,  37.0 ], [ 366.0, 174.0 ], [ 115.0,  59.0 ], \
                 [ 162.0,  87.0 ], [  38.0,  90.0 ], [ 258.0, 173.0 ], \
                 [ 224.0, 108.0 ], [  78.0, 170.0 ], [  72.0,  43.0 ]])] * H * W, \
            (H, W, B, 2)
        )
    center_x = np.reshape(\
            np.transpose(\
                np.reshape(\
                    np.array([np.arange(1, W+1)*float(1248)/(W+1)]*H*B), \
                    (B, H, W)\
                ),\
                (1, 2, 0)\
            ),
            (H, W, B, 1)
        )
    center_y = np.reshape(\
            np.transpose(\
                np.reshape(\
                    np.array([np.arange(1, H+1)*float(384)/(H+1)]*W*B), \
                    (B, W, H)\
                ),\
                (2, 1, 0)\
            ),
            (H, W, B, 1)
        )

    anchors = np.reshape(\
                np.concatenate((center_x, center_y, anchor_shapes), axis = 3), \
                (-1, 4)
        )

    return anchors

def bbox_transform(bbox):
    """
    Args: 
        bbox [BATCH, 4] (centerx, centery, with, height)
    Output:
        out_box: [ABTCH, 4] (xmin, ymin, xmax, ymax)
    """
    cx, cy, w, h = bbox
    out_box = [[]] * 4
    out_box[0] = cx - w / 2
    out_box[1] = cy - h / 2
    out_box[2] = cx + w / 2
    out_box[3] = cy + h / 2
    
    return out_box

def bbox_transform_inv(bbox):
    """
    Args:
        bbox: [BATCH_SIZE, 4], (xmin, ymin, xmax, ymax)
    Output:
        out_box: [BATCH_SIZE, 4], (center_x, center_y, width, height)
    """
    xmin, ymin, xmax, ymax = bbox
    out_box = [[]] * 4
    width = xmax - xmin + 1.0
    height = ymax - ymin + 1.0

    out_box[0] = xmin + 0.5 * width
    out_box[1] = ymin + 0.5 * height
    out_box[2] = width
    out_box[3] = height

    return out_box

def clip_box(bbox):
    """
    Clip bbox larger than image size
    """
    xmin, ymin, xmax, ymax = bbox
    xmin = np.maximum(0.0, np.minimum(xmin, cfg.IMAGE_WIDTH - 1))
    ymin = np.maximum(0.0, np.minimum(ymin, cfg.IMAGE_HEIGHT - 1))
    xmax = np.maximum(0.0, np.minimum(xmax, cfg.IMAGE_WIDTH - 1))
    ymax = np.maximum(0.0, np.minimum(ymax, cfg.IMAGE_HEIGHT - 1))
    
    return xmin, ymin, xmax, ymax

def iou(box1, box2):
    lr = min(box[1]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
        max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    if lr > 0:
        tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
            max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
        if tb > 0:
            intersection = tb * lr
            union = box1[2]*box1[3]+box[2][3]-intersection
            
            return intersection / union
    return 0

def batch_iou(boxes, box):
    lr = np.maximum(\
            np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
            np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]), \
            0
    )
    tb = np.maximum(\
            np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
            np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]), \
            0
    )

    inter = lr * tb
    union = boxes[:,2] * boxes[:,3] + box[2] * box[3] - inter

    return inter / union


def nms(boxes, probs, threshold):
    """Non-Maximimum supression here. Not good
    Args:
        boxes: [BATCH, 4], (cx, cy, w, h)
        probs: [BATCH, 3]
        threshold
    Return:
        keep: array of True or False
    """
    order = probs.argsort()[::-1]
    keep = [True] * len(order)

    for i in range(len(order) - 1):
        ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > threshold:
                keep[order[j+i+1]] = False
    return keep

