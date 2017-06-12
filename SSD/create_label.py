import os
import matplotlib.image as mpimg
import vanilla_ssd as van
import pickle
import cv2
import tensorflow as tf

from datasets import dataset_factory
from preprocessing import preprocessing_factory
import ssd_shrink_network as ssd_s

#data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train'
#output_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC_train/label'

data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/img'
output_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/label'
logfile = '/home/ubuntu/workspace/fastcnn/log/test'
v = van.vanilla_ssd_net('/gpu:1')


batch_size = 1

#data_path = '/home/dashmoment/dataset/demo/img'
#output_path = '/home/dashmoment/dataset/demo/label'
#ckpt_filename = '/home/dashmoment/dataset/model/ssd_300/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
s = ssd_s.ssd_shrink_network('ssd_s08', 0.1,  1, '', '/gpu:1')

image_list = os.listdir(data_path)

fcount = 0

import numpy as np


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
        
        for i in range(len(glabel)):
            
            bbox = glocation[i]
            label = glabel[i]
            prior_scaling=[0.1, 0.1, 0.2, 0.2]
            
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

for fid in range(0,len(image_list)):
    
    fcount = fcount + 1
    
    print("Process:{}/{}".format(fid, len(image_list)))
    
    img_path = os.path.join(data_path, image_list[fid])
    out_file =  os.path.join(output_path, image_list[fid]+'.pickle')
    img = cv2.imread(img_path)
    
    glabel, glocation, gscore = v.inference(img)
    target_labels, target_localizations, target_scores = encode_box(v.ssd_anchors, glabel, glocation)
       
    fglabel, fglocation, fgscore = v.sess.run(v.flatten_output(target_labels, target_localizations, target_scores))
    
    img = s.img_preprocessing(img)
    for i in range(10):
        
        s.sess.run(s.solver, feed_dict={s.inputs:img, s.glabel:fglabel , s.glocation:fglocation, s.gscore:fgscore})
#        print(loss)
    
#    tf.summary.scalar("train_RMSE",s.loss, collections=['train'])
#    merged_summary_train = tf.summary.merge_all('train')
#    
#    summary_writer = tf.summary.FileWriter(logfile, s.sess.graph) 
#    sess = tf.Session()
#    sess.run(tf.global_variables_initializer()) 
#    sum_test = sess.run(merged_summary_train, feed_dict={s.inputs:img, s.glabel:fglabel , s.glocation:fglocation, s.gscore:fgscore})
#    summary_writer.add_summary(sum_test, 0)
#     
#    with open(out_file, 'wb') as f:
#        pickle.dump([fglabel, fglocation, fgscore], f, protocol=pickle.HIGHEST_PROTOCOL)


















