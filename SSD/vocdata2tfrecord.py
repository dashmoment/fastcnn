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



###Code Reference http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

#data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/img'
#tfrecords_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/tfrecord'

data_path =  '/home/dashmoment/dataset/demo/img'
tfrecords_path = '/home/dashmoment/dataset/demo/tfrecord/'
#v = van.vanilla_ssd_net('/gpu:0','/home/dashmoment/dataset/model/ssd_300/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt')
#v = van.vanilla_ssd_net('/gpu:0')

batch_size = 8
s = ssd_s.ssd_shrink_network('ssd_s08', 0.1,  batch_size, '', '/gpu:0')

#def _bytes_feature(value):
#    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#def _int64_feature(value):
#    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
#SIZE_PER_FILE = 10
#
#filelist = os.listdir(data_path)
#
#for i in range(len(filelist)//SIZE_PER_FILE):
#    
#    tfrecords_filename = os.path.join(tfrecords_path, 'voc_for_ssd_training_'+str(i)+'.tfrecords')
#    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
#    
#    print("Process:{}/{}".format(i+1,len(filelist)//SIZE_PER_FILE))
#    
#    start = time.clock()
#    for j in range(SIZE_PER_FILE):
#        
#              
#        file_idx = i*SIZE_PER_FILE + j
#        
#        
#        img_path = os.path.join(data_path, filelist[file_idx])
#        img = cv2.imread(img_path)
#        
#        height = img.shape[0]
#        width = img.shape[1]
#        
#        image_pro, fglabel, fglocation, fgscore = s.van.create_img_label(img)
#        
#        
#        img_raw = image_pro.tostring()
#        fglabel_b = fglabel.tostring()
#        fglocation_b = fglocation.tostring()
#        fgscore_b = fgscore.tostring()
#        
#        example = tf.train.Example(features=tf.train.Features(feature={
#                
#                'height':_int64_feature(height),
#                'width':_int64_feature(width),
#                'imgraw':_bytes_feature(img_raw),
#                'fglabel':_bytes_feature(fglabel_b),
#                'fglocation':_bytes_feature(fglocation_b),
#                'fgscore':_bytes_feature(fgscore_b)
#                }))
#        
#        writer.write(example.SerializeToString())
#    writer.close()
#    
#    end = time.clock()
#    print("time:{}".format(end-start))
    
    
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
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=300,
                                           target_width=300)
    
    images, tfglabel, tfglocation, tfgscore, theight, twidth = tf.train.shuffle_batch( [resized_image ,tfglabel, tfglocation, tfgscore, height, width],
                                         batch_size=batch_size,
                                         capacity=60000,
                                         num_threads=10,
                                         min_after_dequeue=0)
    
    
    
    
    return images, tfglabel, tfglocation, tfgscore, [theight, twidth]
    


#data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/img'
#filelist = os.listdir(data_path)
#img = cv2.imread(os.path.join(data_path, filelist[10]))
#pre_img = s.img_preprocessing(img)
#s.plot(img)
    
tfrecord_list =  os.listdir(tfrecords_path)

for i in range(len(tfrecord_list)) :
    tfrecord_list[i] = os.path.join(tfrecords_path, tfrecord_list[i])

filename_queue = tf.train.string_input_producer(
    tfrecord_list, num_epochs=None)   

img, label, loc, score, img_size  = read_and_decode(filename_queue) 
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())


with tf.Session()  as sess:
    
#    summary_writer = tf.summary.FileWriter('testlog', sess.graph) 
    
    sess.run(tf.local_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)  
    
    try:
        i= 0 
        while not coord.should_stop():
            
            
            i = i + 1
            img_s, tfglabel_s, loc_s, score_s, sizes = sess.run([img, label, loc, score, img_size])
            
            tfglabel_s = np.reshape(tfglabel_s, [-1])
            loc_s = np.reshape(loc_s, [-1,4])
            score_s = np.reshape(score_s, [-1])
            
#            img_re = cv2.resize(img_s[0], (sizes[0],sizes[1]))
#            res_v = s.van.inference(img)
            
#            pre_img = s.img_preprocessing(img_re)
            res_s = s.inference(img_s)
#            loss, _ = s.sess.run([s.loss, s.solver], feed_dict={s.inputs:img_s, s.glabel:tfglabel_s,  s.glocation:loc_s , s.gscore:score_s})
#            s.sess.run(s.loss, feed_dict={s.inputs:img_s, s.glabel:tfglabel_s,  s.glocation:loc_s , s.gscore:score_s})

#            print(loss)

            print("Iteration:{}".format(i))
                       
            
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    
    coord.request_stop()
    coord.join(threads)


#data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/img'
#filelist = os.listdir(data_path)
#img = cv2.imread(os.path.join(data_path, filelist[10]))
#pre_img = s.img_preprocessing(img)
#s.plot(img)
#np.array_equal(res_s[0][0], res_v[0][0])



















    
    
    
    
    
    