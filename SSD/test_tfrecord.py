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

###Code Reference http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

#data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/img'
data_path =  '/home/dashmoment/dataset/demo/img'
tfrecords_filename = '/home/dashmoment/dataset/demo/tfrecord/pascal_voc_segmentation.tfrecords'

v = van.vanilla_ssd_net('/gpu:0','/home/dashmoment/dataset/model/ssd_300/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt')

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


#writer = tf.python_io.TFRecordWriter(tfrecords_filename)
#
#filelist = os.listdir(data_path)
#for i in range(len(filelist)):
#
#
#    img_path = os.path.join(data_path, filelist[0])
#    img = np.array(Image.open(img_path))
#    
#    height = img.shape[0]
#    width = img.shape[1]
#    
#    glabel, glocation, gscore = v.inference(img)
#    target_labels, target_localizations, target_scores = encode_box(v.ssd_anchors, glabel, glocation)      
#    fglabel, fglocation, fgscore = v.sess.run(v.flatten_output(target_labels, target_localizations, target_scores))
#    
#    img_raw = img.tostring()
#    fglabel_b = fglabel.tostring()
#    fglocation_b = fglocation.tostring()
#    fgscore_b = fgscore.tostring()
#    
#    example = tf.train.Example(features=tf.train.Features(feature={
#            
#            'height':_int64_feature(height),
#            'width':_int64_feature(width),
#            'imgraw':_bytes_feature(img_raw),
#            'fglabel':_bytes_feature(fglabel_b),
#            'fglocation':_bytes_feature(fglocation_b),
#            'fgscore':_bytes_feature(fgscore_b)
#            }))
#    
#    writer.write(example.SerializeToString())
#writer.close()


def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'imgraw': tf.FixedLenFeature([], tf.string),
        'fglabel': tf.FixedLenFeature([], tf.string),
        'fglocation': tf.FixedLenFeature([], tf.string),
        'fgscore': tf.FixedLenFeature([], tf.string)
        })
    
    image = tf.decode_raw(features['imgraw'], tf.uint8)
    tfglabel = tf.decode_raw(features['fglabel'], tf.int64)
    tfglocation = tf.decode_raw(features['fglocation'], tf.float32)
    tfgscore = tf.decode_raw(features['fgscore'], tf.float32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    tfglabel = tf.reshape(tfglabel, [8732])
    tfglocation = tf.reshape(tfglocation, [8732,4])
    tfgscore = tf.reshape(tfgscore, [8732])
    
#    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
#                                           target_height=300,
#                                           target_width=300)
    
    resized_image = v.img_preprocessing(image)
    
#    height = tf.cast(features['height'], tf.int32) 
#    width = tf.cast(features['width'], tf.int32)
    
    images, tfglabel, tfglocation, tfgscore = tf.train.shuffle_batch( [resized_image,tfglabel, tfglocation, tfgscore],
                                         batch_size=6,
                                         capacity=60000,
                                         num_threads=10,
                                         min_after_dequeue=0)

#    images= tf.train.shuffle_batch( [resized_image],
#                                         batch_size=13,
#                                         capacity=60000,
#                                         num_threads=10,
#                                         min_after_dequeue=0)
    
    return images, tfglabel, tfglocation, tfgscore

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=175)   

img, label, loc, score  = read_and_decode(filename_queue) 
#img  = read_and_decode(filename_queue) 
    
    
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())


with tf.Session()  as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)  
    
    
    try:
        i= 0 
        while not coord.should_stop():
            
            i = i + 1
            img_s, tfglabel_s, loc_s, score_s = sess.run([img, label, loc, score])

            print("{}/{}".format(i,100))
            
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
    # When done, ask the threads to stop.
        coord.request_stop()
    
    coord.request_stop()
    coord.join(threads)
    
    
    
    
    
    
    
    
    
    
    
    