import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import sys
sys.path=['', '/home/kyxu/anaconda2/envs/tf1.10.0_py3/lib/python35.zip', '/home/kyxu/anaconda2/envs/tf1.10.0_py3/lib/python3.5', '/home/kyxu/anaconda2/envs/tf1.10.0_py3/lib/python3.5/plat-linux', '/home/kyxu/anaconda2/envs/tf1.10.0_py3/lib/python3.5/lib-dynload', '/home/kyxu/.local/lib/python3.5/site-packages', '/home/kyxu/anaconda2/envs/tf1.10.0_py3/lib/python3.5/site-packages']
from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

model_path = "/home/kyxu/lane/lanenet-lane-detection-master/model/tusimple_lanenet_2/tusimple_lanenet_vgg_2019-04-04-11-56-55.ckpt-180"


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = 'lanenet_model/pix_embedding_relu,lanenet_model/binary_seg_argmax' #, lanenet_model/ArgMax
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    #saver = tf.train.Saver()
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    #reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
    #var_to_shape_map = reader.get_variable_to_shape_map()
    #for key in var_to_shape_map:
    #    print('tensor_name: ',key)

    #for node in tf.get_default_graph().as_graph_def().node:
    #    print(node)
 
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/', sess.graph)
        saver.restore(sess=sess, save_path=input_checkpoint)
        #saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

# 输入ckpt模型路径
#input_checkpoint='model/tusimple_lanenet/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000'
input_checkpoint='/home/kyxu/lane/lanenet-lane-detection-master/model/save/lanenet.ckpt-1000'
# 输出pb模型的路径
out_pb_path="/home/kyxu/lane/lanenet-lane-detection-master/model/pb_model2/frozen_model.pb"
# 调用freeze_graph将ckpt转为pb
freeze_graph(input_checkpoint,out_pb_path)

