import os
import os.path as ops
import argparse
import time
import math

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

print(tf.__version__)
print(tf.__path__)

VGG_MEAN = [103.939, 116.779, 123.68]

def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def test_lanenet(image_path, weights_path, use_gpu):
    """
    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    log.info('开始读取图像数据并进行预处理')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image - VGG_MEAN
    log.info('图像读取完毕, 耗时: {:.5f}s'.format(time.time() - t_start))

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(weights_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        # gpu_options = tf.GPUOptions(visible_device_list="1", per_process_gpu_memory_fraction=0.5)
        # sess_config = tf.ConfigProto(gpu_options=gpu_options)
        # serialized = sess_config.SerializeToString()
        # print(list(map(hex, serialized)))
        # print("##################")

        tf.device('/cpu:0')
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        serialized = sess_config.SerializeToString()
        print(list(map(hex, serialized)))
        print("##################")

        sess = tf.Session(config=sess_config)
        with sess.as_default():
            writer = tf.summary.FileWriter('logs/', sess.graph)
            sess.run(tf.global_variables_initializer())
            input_image_tensor = sess.graph.get_tensor_by_name("input_tensor:0")
            # 定义输出的张量名称
            #output_tensor_name = sess.graph.get_tensor_by_name("InceptionV3/Logits/SpatialSqueeze:0")
            binary_seg_ret = sess.graph.get_tensor_by_name("lanenet_model/binary_seg_argmax:0")
            instance_seg_ret = sess.graph.get_tensor_by_name("lanenet_model/pix_embedding_relu:0")
            # 读取测试图片
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字（需要在名字后面加：0），不是操作节点的名字
            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                feed_dict={input_image_tensor: [image]})
            t_cost = time.time() - t_start
        log.info('单张图像车道线预测耗时: {:.5f}s'.format(t_cost))

        binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])

        for i in range(4):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

    sess.close()
    return

test_lanenet("/home/xuky/works/program-libs/lanenet-lane-detection/data/tusimple_test_image/2.jpg", 
    "/home/xuky/works/program-libs/lanenet-lane-detection/model/pb_model-with-input/frozen_model.pb",
    True)