import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

import sys

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing


slim = tf.contrib.slim
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


net_shape = (300, 300)
data_format = 'NHWC' # N image, Height, Width, Channels
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

image_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages/2007_000042.jpg'

#ssd_anchors = ssd_net.anchors(net_shape)

img = cv2.imread(image_path)
print(img.shape)

#image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
#    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)

image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(img_input, None, None,net_shape,data_format)
        
image_4d = tf.expand_dims(image_pre, 0)

isess.run(tf.global_variables_initializer())
img2 = isess.run(image_4d,feed_dict={img_input: img})

