import numpy as np
import tensorflow as tf
import random

import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from nets import custom_layers

from datasets import dataset_factory
from preprocessing import preprocessing_factory



class vanilla_ssd_net:
    
    def __init__(self, gpu = '/gpu:0', ckpt_filename = '/home/ubuntu/workspace/fastcnn/model/SSD_300/ssd_300_vgg.ckpt', reuse=None):
        
        self.gpu = gpu
        self.net_shape = (300, 300)
        self.data_format = 'NHWC'
        self.ckpt_filename = ckpt_filename
        self.reuse = reuse
        
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        self.glabel = tf.placeholder(tf.int64, shape = (None))
        self.glocation = tf.placeholder(tf.float32, shape = (None, 4))
        self.gscore = tf.placeholder(tf.float32)
#        self.inputs = tf.placeholder(tf.float32, shape=(None, self.net_shape[0], self.net_shape[1], 3))
        
        self.build_model()
        
        
    
    def build_model(self):
        
        with tf.device(self.gpu):
            
         # Evaluation pre-processing: resize to SSD net shape.
            image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            self.image_4d = tf.expand_dims(image_pre, 0)
            
#            reuse = True if 'ssd_net' in locals() else None
            self.ssd_net = ssd_vgg_300.SSDNet()
            
             # SSD default anchor boxes.
            self.ssd_anchors = self.ssd_net.anchors(self.net_shape)
            
            
            with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
                self.predictions, self.localisations,  self.logits, _ = self.ssd_net.net(self.image_4d, is_training=False, reuse=self.reuse)
            
            self.sess = tf.Session(config=self.config)
            self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.ckpt_filename)
        
#        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''):
#            print (i.name)
    

    def img_preprocessing(self, img):
        image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        image_4d = tf.expand_dims(image_pre, 0)
        
        pre_img = self.sess.run(image_4d, feed_dict={self.img_input:img})
        
        return pre_img
    
        
    def inference(self,img):
        
        
        img_pro, rlogit, self.rpredictions, self.rlocalisations, self.rbbox_img = self.sess.run([self.img_input, self.logits, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                    self.rpredictions, self.rlocalisations, self.ssd_anchors,
                    select_threshold=0.5, img_shape=self.net_shape, num_classes=21, decode=True)
        
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)
        
        rbboxes = np_methods.bboxes_resize(self.rbbox_img, rbboxes)
        
    #    return rlogit
        return img_pro ,rclasses, rbboxes, rscores

#        return self.rpredictions, self.rlocalisations, rscores
       
    
    def create_img_label(self, img):
        
        img_pro, rlogit, self.rpredictions, self.rlocalisations, self.rbbox_img = self.sess.run([self.img_input, self.logits, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                    self.rpredictions, self.rlocalisations, self.ssd_anchors,
                    select_threshold=0.5, img_shape=self.net_shape, num_classes=21, decode=True)
        
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)
        
        rbboxes = np_methods.bboxes_resize(self.rbbox_img, rbboxes)
        
        target_labels, target_localizations, target_scores = encode_box(self.ssd_anchors, rclasses, rbboxes)
        
        fglabel, fglocation, fgscore = self.sess.run(self.flatten_output(target_labels, target_localizations, target_scores))
        
        return img_pro, fglabel, fglocation, fgscore
        
    def flatten_output(self, glabel, glocation, gscore):

        # Flatten out all vectors!
        
        fgclasses = []
        fgscores = []
        fglocalisations = []
    
        for i in range(len(glabel)):
            
            fgclasses.append(tf.reshape(glabel[i], [-1]))
            fgscores.append(tf.reshape(gscore[i], [-1]))          
            fglocalisations.append(tf.reshape(glocation[i].astype(np.float32), [-1, 4]))
            
        
        gclasses = tf.concat(fgclasses, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        
        return gclasses, glocalisations, gscores
    
    def predict_box(self, img):
        
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        self.rpredictions, self.rlocalisations, self.ssd_anchors,
        select_threshold=0.5, img_shape=self.net_shape, num_classes=21, decode=True)

        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)
        
        rbboxes = np_methods.bboxes_resize(self.rbbox_img, rbboxes)
        
        return rclasses, rscores, rbboxes
        
        

    def plot(self, img):

        img_pro, rlogit, self.rpredictions, self.rlocalisations, self.rbbox_img = self.sess.run([self.img_input, self.logits, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        self.rpredictions, self.rlocalisations, self.ssd_anchors,
        select_threshold=0.5, img_shape=self.net_shape, num_classes=21, decode=True)

        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(self.rbbox_img, rbboxes)
        plt_bboxes(img, rclasses, rscores, rbboxes)
        
        
    def box_encode(self, glabels, gbboxes):      
        gclasses, glocalisations, gscores = \
                    self.ssd_net.bboxes_encode(glabels, gbboxes, self.ssd_anchors)
                    
        return gclasses, glocalisations, gscores
        
def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    
    """
    classes_label =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
     
    
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
            class_name = str(classes_label[cls_id])
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()


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
