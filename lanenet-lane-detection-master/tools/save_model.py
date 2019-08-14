import os
import os.path as ops
import argparse
import time
import math
import sys
sys.path=['', '/home/kyxu/anaconda2/envs/tf1.10.0_py3/lib/python35.zip', '/home/kyxu/anaconda2/envs/tf1.10.0_py3/lib/python3.5', '/home/kyxu/anaconda2/envs/tf1.10.0_py3/lib/python3.5/plat-linux', '/home/kyxu/anaconda2/envs/tf1.10.0_py3/lib/python3.5/lib-dynload', '/home/kyxu/.local/lib/python3.5/site-packages', '/home/kyxu/anaconda2/envs/tf1.10.0_py3/lib/python3.5/site-packages']
import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def test_lanenet(weights_path, save_path):

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    for node in tf.get_default_graph().as_graph_def().node:
        print(node)
    #print(tf.global_variables())
    
    saver = tf.train.Saver()

    # Set sess configuration
    #tf.device('/gpu:0')
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    #print("use_gpu:", use_gpu)
    #if use_gpu:
    #    sess_config = tf.ConfigProto(device_count={'GPU': 0})
    #    print("use_gpu")
    #else:
    #    sess_config = tf.ConfigProto(device_count={'CPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        writer = tf.summary.FileWriter('logs/', sess.graph)

        saver.restore(sess=sess, save_path=weights_path)
        saver.save(sess, save_path, global_step=1000)

    sess.close()
    return

# 输入ckpt模型路径
input_checkpoint='/home/kyxu/lane/lanenet-lane-detection-master/model/tusimple_lanenet_2/tusimple_lanenet_vgg_2019-04-04-11-56-55.ckpt-180'
# 输出pb模型的路径
out_pb_path="/home/kyxu/lane/lanenet-lane-detection-master/model/save/lanenet.ckpt"
# 调用freeze_graph将ckpt转为pb
test_lanenet(input_checkpoint,out_pb_path)
