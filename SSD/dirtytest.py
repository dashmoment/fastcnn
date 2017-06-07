import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from nets import custom_layers

import utility as ut
import vanilla_ssd as van
import ssd_shrink_network as ssd_s

img_shape=(300, 300)
num_classes=21
no_annotation_label=21
feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
anchor_size_bounds=[0.15, 0.90]
# anchor_size_bounds=[0.20, 0.90],
anchor_sizes=[(21., 45.),
              (45., 99.),
              (99., 153.),
              (153., 207.),
              (207., 261.),
              (261., 315.)]

anchor_ratios=[[2, .5],
               [2, .5, 3, 1./3],
               [2, .5, 3, 1./3],
               [2, .5, 3, 1./3],
               [2, .5],
               [2, .5]]
anchor_steps=[8, 16, 32, 64, 100, 300]
anchor_offset=0.5
normalizations=[20, -1, -1, -1, -1, -1]
prior_scaling=[0.1, 0.1, 0.2, 0.2]
prediction_fn=slim.softmax


dropout_keep_prob = 0.5
reuse = None
is_training = True
scope_name  = 'test'

ratio = 0.8

v = van.vanilla_ssd_net()
#s = ssd_s.ssd_shrink_network(scope_name, ratio)


# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''):
            print (i.name)


#ssd_s.model_prunning('ssd_300_vgg', scope_name, v,s)
##
#####-------------Prunung Weight Test-------------------
#
#with tf.variable_scope(scope_name) as scope:
#    scope.reuse_variables()
#    va = tf.get_variable('block8/conv1x1/weights')
#    
#tensors_s = s.sess.run(va)
#with tf.variable_scope('ssd_300_vgg') as scope:
#    scope.reuse_variables()
#    va2= tf.get_variable('block8/conv1x1/weights')
#    
#tensors = v.sess.run(va2) 

## SSD default anchor boxes.
#ssd_anchors = ssd_net.anchors(net_shape)


#path = '../demo/'
#image_names = sorted(os.listdir(path))
#image_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages/2010_001884.jpg'
image_path ='/home/dashmoment/dataset/demo/000001.jpg'
img = mpimg.imread(image_path)
#glabel, glocation, gscore = v.inference(img)

ssd_net = ssd_vgg_300.SSDNet()
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config= config)
sess.run(tf.global_variables_initializer())


inputa = sess.run(image_4d, feed_dict={img_input:img})
#fglabel, fglocation, fgscore = s.flatten_output(glabel, glocation, gscore)

#for i in range(10):
#    s.train_op(img, glabel, glocation, gscore )
#
#v.plot(img)


























