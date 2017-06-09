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
import tf_utils

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
s = ssd_s.ssd_shrink_network(scope_name, ratio,2)


# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
#img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
## Evaluation pre-processing: resize to SSD net shape.
#image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
#    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
#image_4d = tf.expand_dims(image_pre, 0)

ssd_net = ssd_vgg_300.SSDNet()
ssd_anchors = ssd_net.anchors(net_shape)
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



#image_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages/2010_001884.jpg'
#image_path2 = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages/2011_003208.jpg'
image_path ='/home/dashmoment/dataset/demo/000001.jpg'
image_path2 ='/home/dashmoment/dataset/demo/000001.jpg'

img = mpimg.imread(image_path)
img2 = mpimg.imread(image_path2)



img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config= config)
sess.run(tf.global_variables_initializer())


inputa = sess.run(image_4d, feed_dict={img_input:img})
inputa2 = sess.run(image_4d, feed_dict={img_input:img2})

inp = [inputa,inputa2]
a = np.vstack(inp)

glabel, glocation, gscore = v.inference(img)
fglabel, fglocation, fgscore = s.sess.run(s.flatten_output(glabel, glocation, gscore))
glabel2, glocation2, gscore2 = v.inference(img2)
fglabel2, fglocation2, fgscore2 = s.sess.run(s.flatten_output(glabel2, glocation2, gscore2))
#p = s.sess.run(s.logits, feed_dict={s.inputs:a})
#
fglabel = np.reshape(np.stack([fglabel, fglabel2]),[-1])
fglocation = np.reshape(np.stack([fglocation, fglocation2]),[-1,4])
fgscore = np.reshape(np.stack([fgscore, fgscore2]),[-1])

for i in range(10):
    _, loss = s.sess.run([s.solver, s.loss],  feed_dict={s.inputs: a , s.glabel:fglabel, s.glocation:fglocation, s.gscore:fgscore})


#s.train_op(a, fglabel, fglocation, fgscore )
#g_label = []
#g_location = []
#g_score = []
#
#for i in range(len(glabel)):    
#    
#    tmp = np.stack((glabel[i], glabel2[i]))
#    tmpl = np.stack((glocation[i], glocation2[i]))
#    tmps = np.stack((gscore[i], gscore2[i]))
#    g_label.append(tmp)
#    g_location.append(tmpl)
#    g_score.append(tmps)

#fglabel, fglocation, fgscore = s.flatten_output(glabel, glocation, gscore)


#for i in range(10):
#    s.train_op(a, glabel, glocation, gscore )

#v.plot(img)

#from datasets import dataset_factory
#from preprocessing import preprocessing_factory
#
##with tf.device('/gpu:1'):
#    
#tfreader = tf.TFRecordReader()
#
#filenames = ['/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012_record/voc_2012_train_000.tfrecord']
#filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
#reader = tfreader.read(filename_queue) 
#dataset = dataset_factory.get_dataset(
#            'pascalvoc_2012', 'train', '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012_record')
#
#batch_size = 64
#batch_shape = [1] + [len(ssd_anchors)] * 3
#
#provider = slim.dataset_data_provider.DatasetDataProvider(
#                    dataset,
#                    num_readers=10,
#                    common_queue_capacity=20*batch_size ,
#                    common_queue_min=10*batch_size,
#                    shuffle=True)
#[image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
#                                                         'object/label',
#                                                         'object/bbox'])
#
#image_preprocessing_fn = preprocessing_factory.get_preprocessing(
#        'ssd_300_vgg', is_training=True)
#
#
#    
#image, glabels, gbboxes = \
#            image_preprocessing_fn(image, glabels, gbboxes,
#                                   out_shape=(300,300),
#                                   data_format=data_format)
#            
#gclasses, glocalisations, gscores = \
#                ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
#                
#r = tf.train.batch(
#            tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
#            batch_size=batch_size,
#            num_threads=10,
#            capacity=5 *batch_size)
#
#b_image, b_gclasses, b_glocalisations, b_gscores = \
#                tf_utils.reshape_list(r, batch_shape)
#
#
#
# 
#with tf.Session() as sess:
#    
##    for i in range(10):
#    
#    
#    sess.run(tf.local_variables_initializer())
#    sess.run(tf.global_variables_initializer())
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#
#    a = sess.run([b_gclasses,b_glocalisations, b_gscores, b_image])
#    
#    arg_scope = ssd_net.arg_scope(weight_decay=0.8,
#                                          data_format=data_format)
#    with slim.arg_scope(arg_scope):
#        predictions, localisations, logits, end_points = \
#            ssd_net.net(b_image, is_training=True)
#    # Add loss function.
#
#    coord.request_stop()
#    coord.join(threads)
#
#
#glabel , glocation, gscore = s.sess.run(s.flatten_output(a[0],a[1],a[2]))    
#r,_ = s.sess.run([s.loss, s.solver], feed_dict={s.inputs:a[3], s.glabel:glabel, s.glocation:glocation, s.gscore:gscore})

















