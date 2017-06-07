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

init_layers = {
            'conv1/conv1_1/weights':64,
            'conv1/conv1_1/biases':64,
            'conv1/conv1_2/weights':64,
            'conv1/conv1_2/biases':64,
            'conv2/conv2_1/weights':128,
            'conv2/conv2_1/biases':128,
            'conv2/conv2_2/weights':128,
            'conv2/conv2_2/biases':128,
            'conv3/conv3_1/weights':256,
            'conv3/conv3_1/biases':256,
            'conv3/conv3_2/weights':256,
            'conv3/conv3_2/biases':256,
            'conv3/conv3_3/weights':256,
            'conv3/conv3_3/biases':256,
            'conv4/conv4_1/weights':512,
            'conv4/conv4_1/biases':512,
            'conv4/conv4_2/weights':512,
            'conv4/conv4_2/biases':512,
            'conv4/conv4_3/weights':512,
            'conv4/conv4_3/biases':512,
            'conv5/conv5_1/weights':512,
            'conv5/conv5_1/biases':512,
            'conv5/conv5_2/weights':512,
            'conv5/conv5_2/biases':512,
            'conv5/conv5_3/weights':512,
            'conv5/conv5_3/biases':512,
            'conv6/weights':1024,
            'conv6/biases':1024,
            'conv7/weights':1024,
            'conv7/biases':1024,
            'block8/conv1x1/weights':256,
            'block8/conv1x1/biases':256,
            'block8/conv3x3/weights':512,
            'block8/conv3x3/biases':512,
            'block9/conv1x1/weights':128,
            'block9/conv1x1/biases':128,
            'block9/conv3x3/weights':256,
            'block9/conv3x3/biases':256,
            'block10/conv1x1/weights':128,
            'block10/conv1x1/biases':128,
            'block10/conv3x3/weights':256,
            'block10/conv3x3/biases':256,
            'block11/conv1x1/weights':128,
            'block11/conv1x1/biases':128,
            'block11/conv3x3/weights':256,
            'block11/conv3x3/biases':256
        }


def creat_network(scope, ratio, inputs, ssd_obj):

    with slim.arg_scope(ssd_obj.arg_scope(data_format=data_format)):
        end_points = {}
        with tf.variable_scope(scope,scope,[inputs], reuse=reuse):
            # Original VGG-16 blocks.
            net = slim.repeat(inputs, 2, slim.conv2d, int(64*ratio), [3, 3], scope='conv1')
            end_points['block1'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            # Block 2.
            net = slim.repeat(net, 2, slim.conv2d, int(128*ratio), [3, 3], scope='conv2')
            end_points['block2'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # Block 3.
            net = slim.repeat(net, 3, slim.conv2d, int(256*ratio), [3, 3], scope='conv3')
            end_points['block3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            # Block 4.
            net = slim.repeat(net, 3, slim.conv2d, int(512*ratio), [3, 3], scope='conv4')
            end_points['block4'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            # Block 5.
            net = slim.repeat(net, 3, slim.conv2d, int(512*ratio), [3, 3], scope='conv5')
            end_points['block5'] = net
            net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')
        
            # Additional SSD blocks.
            # Block 6: let's dilate the hell out of it!
            net = slim.conv2d(net, int(1024*ratio), [3, 3], rate=6, scope='conv6')
            end_points['block6'] = net
            net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
            # Block 7: 1x1 conv. Because the fuck.
            net = slim.conv2d(net, int(1024*ratio), [1, 1], scope='conv7')
            end_points['block7'] = net
            net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        
            # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, int(256*ratio), [1, 1], scope='conv1x1')
                net = custom_layers.pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, int(512*ratio), [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block9'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, int(128*ratio), [1, 1], scope='conv1x1')
                net = custom_layers.pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, int(256*ratio), [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block10'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, int(128*ratio), [1, 1], scope='conv1x1')
                net = slim.conv2d(net, int(256*ratio), [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block11'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, int(128*ratio), [1, 1], scope='conv1x1')
                net = slim.conv2d(net, int(256*ratio), [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            
            # Prediction and localisations layers.
            predictions = []
            logits = []
            localisations = []
            for i, layer in enumerate(ssd_obj.params.feat_layers):
                with tf.variable_scope(layer + '_box'):
                    p, l = ssd_vgg_300.ssd_multibox_layer(end_points[layer],
                                              num_classes,
                                              anchor_sizes[i],
                                              anchor_ratios[i],
                                              normalizations[i])
                predictions.append(prediction_fn(p))
                logits.append(p)
                localisations.append(l)
                
            return predictions, localisations, logits, end_points


dropout_keep_prob = 0.5
reuse = None
is_training = True
scope_name  = 'test'

ratio = 0.8

v = van.vanilla_ssd_net()
s = ssd_s.ssd_shrink_network(scope_name, ratio)


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


ssd_s.model_prunning('ssd_300_vgg', scope_name, v,s)
#creat_network(scope_name, ratio , image_4d, ssd_net)
##Define the SSD model.


#gpu_options = tf.GPUOptions(allow_growth=True)
#config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
#isess = tf.InteractiveSession(config=config)
#isess.run(tf.global_variables_initializer())

####-------------Prunung Weight Test-------------------

with tf.variable_scope(scope_name) as scope:
    scope.reuse_variables()
    va = tf.get_variable('block8/conv1x1/weights')
    
tensors_s = s.sess.run(va)
with tf.variable_scope('ssd_300_vgg') as scope:
    scope.reuse_variables()
    va2= tf.get_variable('block8/conv1x1/weights')
    
tensors = v.sess.run(va2) 

## SSD default anchor boxes.
#ssd_anchors = ssd_net.anchors(net_shape)


#path = '../demo/'
#image_names = sorted(os.listdir(path))
image_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages/2010_001884.jpg'
img = mpimg.imread(image_path)
glabel, glocation, gscore = v.inference(img)



#r = s.flatten_output(glabel, glocation, gscore)
s.train_op(img, glabel, glocation, gscore )

v.plot(img)







