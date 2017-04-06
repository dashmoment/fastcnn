#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:59:54 2017

@author: ubuntu
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import voc_utils
import YOLO_tiny_tf


cat = voc_utils.list_image_sets()
im = voc_utils.imgs_from_category("train",cat[0])
iml = voc_utils.imgs_from_category_as_list("train",cat[0])

ann = voc_utils.load_annotation(iml[0])
mask = voc_utils.get_masks("train",cat[0],"bbox1")
#fromfile = "test/2008_000090.jpg"
#
#yolo = YOLO_tiny_tf.YOLO_TF()
#
#img = cv2.imread(fromfile)
#
#img_resized = cv2.resize(img, (448, 448))
#img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
#img_resized_np = np.asarray( img_RGB )
#inputs = np.zeros((1,448,448,3),dtype='float32')
#inputs[0] = (img_resized_np/255.0)*2.0-1.0
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    in_dict = {yolo.x: inputs}
#    net_output = yolo.sess.run(yolo.fc_19,feed_dict={yolo.x: inputs})
#
#
#
#yolo.detect_from_file(fromfile)




