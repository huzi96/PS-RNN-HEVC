# Data augmented
from __future__ import print_function
import tensorflow as tf
import cv2
import h5py
import numpy as np
import sys
import os
import subprocess as sp
from tensorflow.python.framework.graph_util import convert_variables_to_constants

batch_size = 1 
epochs = 1000


def tf_build_model(module_name, weights_name, params, input_tensor, output_tensor, mode):
    with tf.variable_scope('main_full', reuse=tf.AUTO_REUSE):
        model_module = __import__(module_name)
        train_op, satd_op, mse_op = model_module.build_model(
            input_tensor, output_tensor, params, mode=mode)
        return train_op, satd_op, mse_op


def drive():
    block_size = 8
    model_module_name = sys.argv[2]
    model_id = sys.argv[3]
    weights_name = None
    mode = int(sys.argv[4])
    num_scale = int(sys.argv[5])
    if len(sys.argv) == 7:
        weights_name = sys.argv[6]
    print(weights_name)
    # load data

    with tf.Session() as sess:
        pass

    inputs = tf.placeholder(tf.float32, [batch_size, num_scale*mode, num_scale*mode, 1])
    targets = tf.placeholder(tf.float32, [batch_size, num_scale, num_scale, 1])

    # build model
    train_op, satd_loss, mse_loss = tf_build_model(model_module_name,
                                       weights_name,
                                       {'learning_rate': 0.0001,
                                           'batch_size': batch_size,
                                           'num_scale':num_scale,
                                           'inner_scale':8},
                                       inputs,
                                       targets,mode)
    
    tensorboard_dir = 'tensorboard'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    checkpoint_dir = './ckpt/'
    with tf.Session() as sess:
        saver.restore(sess, weights_name)
        graph = convert_variables_to_constants(sess, sess.graph_def, ['main_full/conv11/BiasAdd'])
        tf.train.write_graph(graph,'.','graph_m%d_s%d_%s.pb' % (mode, num_scale, model_id),as_text=False)

        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(sess.graph, cmd='op', options=opts)
        if flops is not None:
            print('TF stats gives',flops.total_float_ops)


if __name__ == '__main__':
    tasks = {'train': drive}
    task = sys.argv[1]
    tasks[task]()
