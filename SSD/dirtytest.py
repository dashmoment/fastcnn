import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization


#default_params = SSDParams(
#        img_shape=(300, 300),
#        num_classes=21,
#        no_annotation_label=21,
#        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
#        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
#        anchor_size_bounds=[0.15, 0.90],
#        # anchor_size_bounds=[0.20, 0.90],
#        anchor_sizes=[(21., 45.),
#                      (45., 99.),
#                      (99., 153.),
#                      (153., 207.),
#                      (207., 261.),
#                      (261., 315.)],
#        # anchor_sizes=[(30., 60.),
#        #               (60., 111.),
#        #               (111., 162.),
#        #               (162., 213.),
#        #               (213., 264.),
#        #               (264., 315.)],
#        anchor_ratios=[[2, .5],
#                       [2, .5, 3, 1./3],
#                       [2, .5, 3, 1./3],
#                       [2, .5, 3, 1./3],
#                       [2, .5],
#                       [2, .5]],
#        anchor_steps=[8, 16, 32, 64, 100, 300],
#        anchor_offset=0.5,
#        normalizations=[20, -1, -1, -1, -1, -1],
#        prior_scaling=[0.1, 0.1, 0.2, 0.2]
#        )

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, logit, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

localisations = ssd_vgg_300.tensor_shape(localisations, 4)
    

#Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


path = '../demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-5])

rlogit, rimg, rpredictions, rlocalisations, rbbox_img = isess.run([logit, image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

