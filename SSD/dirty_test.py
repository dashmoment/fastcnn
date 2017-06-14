import os
import matplotlib.image as mpimg
import vanilla_ssd as van
import pickle
import cv2
import tensorflow as tf

from datasets import dataset_factory
from preprocessing import preprocessing_factory
import ssd_shrink_network as ssd_s
import numpy as np
from random import shuffle
import time

from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt


def encode_box(anchors, glabel, glocation, ):
    
    target_labels = []
    target_localizations = []
    target_scores = []
    
    for j in range(len(anchors)):
        yref, xref, href, wref  = anchors[j]
        ymin = yref - href / 2.
        xmin = xref - wref / 2.
        ymax = yref + href / 2.
        xmax = xref + wref / 2.
        vol_anchors = (xmax - xmin) * (ymax - ymin)
        shape = (yref.shape[0], yref.shape[1], href.size)
        
        dtype = np.float32
        feat_labels = np.zeros(shape, dtype=np.int64)
        feat_scores = np.zeros(shape, dtype=dtype)
    
        feat_ymin = np.zeros(shape, np.dtype)
        feat_xmin = np.zeros(shape, dtype=dtype)
        feat_ymax = np.ones(shape, dtype=dtype)
        feat_xmax = np.ones(shape, dtype=dtype)
        
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        
        for i in range(len(glabel)):
            
            bbox = glocation[i]
            label = glabel[i]
            
            
            int_ymin = np.maximum(ymin, bbox[0])
            int_xmin = np.maximum(xmin, bbox[1])
            int_ymax = np.maximum(ymax, bbox[2])
            int_xmax = np.maximum(xmax, bbox[3])
            h = np.maximum(int_ymax - int_ymin, 0.)
            w = np.maximum(int_xmax - int_xmin, 0.)
            inter_vol = h * w
            union_vol = vol_anchors - inter_vol \
                    + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            jaccard = np.divide(inter_vol, union_vol)
            
            mask = np.greater(jaccard, feat_scores)
            mask = np.logical_and(mask, feat_scores > -0.5)
            mask = np.logical_and(mask, label < 21)
            
            imask = mask.astype(np.int64)
            fmask = mask.astype(dtype)
            
            feat_labels = imask * label + (1 - imask) * feat_labels
            feat_scores = np.where(mask, jaccard, feat_scores)
            
            
            feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
            feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
            feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
            feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax
        
        # Transform to center / size.
        feat_cy = (feat_ymax + feat_ymin) / 2.
        feat_cx = (feat_xmax + feat_xmin) / 2.
        feat_h = feat_ymax - feat_ymin
        feat_w = feat_xmax - feat_xmin
        # Encode features.
        feat_cy = (feat_cy - yref) / href / prior_scaling[0]
        feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
        
        tmph = feat_h / href
        feat_h = np.log(tmph.astype(dtype)) / prior_scaling[2]
        tmpw = feat_w / wref
        feat_w = np.log(tmpw.astype(dtype)) / prior_scaling[3]
        feat_localizations = np.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    
        target_labels.append(feat_labels)
        target_localizations.append(feat_localizations)
        target_scores.append(feat_scores)
        
    return target_labels, target_localizations, target_scores

s = ssd_s.ssd_shrink_network('ssd_s08', 1,  1, '', '/gpu:1')
data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/img'
filelist = os.listdir(data_path)
img = cv2.imread(os.path.join(data_path, filelist[10]))


with tf.variable_scope('ssd_300_vgg') as scope:
    scope.reuse_variables()
    var_ori = tf.get_variable('conv3/conv3_3/weights')


a_ori = s.van.sess.run(var_ori)        
        
with tf.variable_scope('ssd_s08') as scope:
    scope.reuse_variables()
    var_s = tf.get_variable('conv3/conv3_3/weights')
a = s.sess.run(var_s)

print(np.array_equal(a, a_ori))


summary_writer = tf.summary.FileWriter('testlog/', s.sess.graph) 

s.van.plot(img)
pre_img = s.img_preprocessing(img)
s.plot(pre_img, img)

pro_img , fglabel, fglocation, fgscore = s.van.create_img_label(img)
tfglabel_s = np.reshape(fglabel, [-1])
loc_s = np.reshape(fglocation, [-1,4])
score_s = np.reshape(fgscore, [-1])
loss = s.sess.run(s.loss, feed_dict={s.inputs:pre_img, s.glabel:fglabel,  s.glocation:loc_s , s.gscore:score_s})

rclasses, rscores, rbboxes = s.inference(pre_img)
target_labels, target_localizations, target_scores = encode_box( s.van.ssd_anchors, rclasses, rbboxes)
tfglabel, tfglocation, tfgscore = s.sess.run(s.flatten_output(target_labels, target_localizations, target_scores))
        

s.plot(pre_img, img)
#res_v = s.van.inference(img)
#


#res_s = s.inference(pre_img)
#res_s = s.sess.run(s.logits, feed_dict={s.img_input:img})

#print(np.array_equal(res_v[0], res_s[0]))