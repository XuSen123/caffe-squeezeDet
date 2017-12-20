# encoding=utf-8
# -----------------------------
# caffe squeezeDet
# Written by XuSenhai
# -----------------------------

"""Set up paths"""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'caffe-master', 'python')
add_path(caffe_path)

lib_path = osp.join(this_dir, '..')

add_path(lib_path)
