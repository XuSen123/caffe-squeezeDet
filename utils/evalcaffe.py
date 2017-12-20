#!/usr/bin/env python
# encoding-utf-8

# ---------------------------------------
# caffe squeezeDet
# Written by XuSenhai
# ---------------------------------------

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import numpy as np
import os
import pdb

absroot = os.getcwd()

def main():
    # Set gpu mode
    caffe.set_mode_gpu()

    # Set net file path
    net_file = os.path.join(absroot, 'prototxt', \
                            'squeezedet.prototxt')

    # Set npy model
    npy_file = os.path.join(absroot, 'model_checkpoints', \
                            'squeezeDet.npy')

    # Set caffe model
    caffe_file = os.path.join(absroot, 'model_checkpoints', \
                             'squeezeDet.caffemodel')

    net = caffe.Net(net_file, caffe_file, caffe.TEST)
    
    paramnames = net.params.keys()
    
    # Load npy model
    npy_model = np.load(npy_file).item()

    for paramname in paramnames:
        npy_weights = npy_model[paramname]['weights']
        npy_biases = npy_model[paramname]['biases']

        caffe_weights = net.params[paramname][0].data
        caffe_biases = net.params[paramname][1].data

        if np.sum(caffe_weights - npy_weights) != 0  or np.sum(caffe_biases - npy_biases) != 0:
            print paramname

if __name__ == '__main__':
    main()
