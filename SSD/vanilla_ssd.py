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


class vanilla_ssd_net:
    
    def __init__(self, gpu = '/gpu:0'):
        
        self.gpu = gpu
        self.net_shape = (300, 300)
        self.data_format = 'NHWC'
        self.ckpt_filename = '/home/ubuntu/workspace/fastcnn/model/SSD_300/ssd_300_vgg.ckpt'
        
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        
        self.build_model()
        
        
    
    def build_model(self):
        
        with tf.device(self.gpu):
        
         # Evaluation pre-processing: resize to SSD net shape.
            image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            self.image_4d = tf.expand_dims(image_pre, 0)
            
            reuse = True if 'ssd_net' in locals() else None
            self.ssd_net = ssd_vgg_300.SSDNet()
            with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
                self.predictions, self.localisations,  self.logits, _ = self.ssd_net.net(self.image_4d, is_training=False, reuse=reuse)
            
            self.sess = tf.Session(config=self.config)
            self.sess.run(tf.global_variables_initializer())
#        self.saver = tf.train.Saver()
#        self.saver.restore(self.sess, self.ckpt_filename)
        
#        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''):
#            print (i.name)
        
        

    def inference(self,img):
        
        
        
        # SSD default anchor boxes.
        self.ssd_anchors = self.ssd_net.anchors(self.net_shape)
        
        rlogit, self.rpredictions, self.rlocalisations, self.rbbox_img = self.sess.run([self.logits, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                    self.rpredictions, self.rlocalisations, self.ssd_anchors,
                    select_threshold=0.5, img_shape=self.net_shape, num_classes=21, decode=True)
        
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)
        
        label = tf.placeholder(tf.int64, shape=(len(rclasses)))
        location = tf.placeholder(tf.float32, shape=(len(rclasses), 4))
        
        tlabel, tlocation, tscore = self.ssd_net.bboxes_encode(label, location, self.ssd_anchors)
        self.glabel, self.glocation, self.gscore = self.sess.run([tlabel, tlocation, tscore] , feed_dict={label:rclasses, location:rbboxes})
        
        return self.glabel, self.glocation, self.gscore
       
       


    def plot(self, img):
        
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        self.rpredictions, self.rlocalisations, self.ssd_anchors,
        select_threshold=0.5, img_shape=self.net_shape, num_classes=21, decode=True)

        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(self.rbbox_img, rbboxes)
        plt_bboxes(img, rclasses, rscores, rbboxes)
        
def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
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



