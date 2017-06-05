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


 


def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()



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

dropout_keep_prob = 0.5
reuse = True
is_training = True
scope  = 'test'

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

#end_points = {}
#with tf.variable_scope(scope, 'ssd_300_vgg', [image_4d], reuse=reuse):
#    # Original VGG-16 blocks.
#    net = slim.repeat(image_4d, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#    end_points['block1'] = net
#    net = slim.max_pool2d(net, [2, 2], scope='pool1')
#    # Block 2.
#    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#    end_points['block2'] = net
#    net = slim.max_pool2d(net, [2, 2], scope='pool2')
#    # Block 3.
#    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#    end_points['block3'] = net
#    net = slim.max_pool2d(net, [2, 2], scope='pool3')
#    # Block 4.
#    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#    end_points['block4'] = net
#    net = slim.max_pool2d(net, [2, 2], scope='pool4')
#    # Block 5.
#    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#    end_points['block5'] = net
#    net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')
#
#    # Additional SSD blocks.
#    # Block 6: let's dilate the hell out of it!
#    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
#    end_points['block6'] = net
#    net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
#    # Block 7: 1x1 conv. Because the fuck.
#    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
#    end_points['block7'] = net
#    net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
#
#    # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
#    end_point = 'block8'
#    with tf.variable_scope(end_point):
#        net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
#        net = custom_layers.pad2d(net, pad=(1, 1))
#        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
#    end_points[end_point] = net
#    end_point = 'block9'
#    with tf.variable_scope(end_point):
#        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
#        net = custom_layers.pad2d(net, pad=(1, 1))
#        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
#    end_points[end_point] = net
#    end_point = 'block10'
#    with tf.variable_scope(end_point):
#        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
#        net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
#    end_points[end_point] = net
#    end_point = 'block11'
#    with tf.variable_scope(end_point):
#        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
#        net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
#    end_points[end_point] = net

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


## Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    prediction, localisations,  logits, end_points = ssd_net.net(image_4d, is_training=False, reuse=reuse)
#
##localisations2 = ssd_vgg_300.tensor_shape(localisations, 4)
#    
#
##Restore SSD model.
#ckpt_filename = '/home/ubuntu/workspace/fastcnn/model/SSD_300/ssd_300_vgg.ckpt'
## ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
#isess.run(tf.global_variables_initializer())
#saver = tf.train.Saver()
#saver.restore(isess, ckpt_filename)
#
## SSD default anchor boxes.
#ssd_anchors = ssd_net.anchors(net_shape)
#
#
##path = '../demo/'
##image_names = sorted(os.listdir(path))
#image_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages/2010_001884.jpg'
#
#img = mpimg.imread(image_path)
#
#rlogit, rimg, rpredictions, rlocalisations, rbbox_img = isess.run([logits, image_4d, prediction, localisations, bbox_img],
#                                                              feed_dict={img_input: img})
#
#rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
#            rpredictions, rlocalisations, ssd_anchors,
#            select_threshold=0.5, img_shape=net_shape, num_classes=21, decode=True)
#
#rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
#rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)
## Resize bboxes to original image shape. Note: useless for Resize.WARP!
#rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
#plt_bboxes(img, rclasses, rscores, rbboxes)







