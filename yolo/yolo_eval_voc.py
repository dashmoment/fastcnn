import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2
import voc_utils as voc
import os
import utility as ut
from bs4 import BeautifulSoup as soup
    

img_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages'
labelfiles = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/ImageSets/Main'
graph_model = '../../model/yolo_ds/fcann_v1.ckpt-62000.meta'
checkpoint_dir = '../../model/yolo_ds'
classes = voc.list_image_sets()
val_list = voc.imgs_from_category_as_list('', 'val', labelfiles)


ds_yolo = {
        'conv1w':tf.Variable(tf.truncated_normal([3,3,3,16], mean=0, stddev=0.01)),
        'conv1b':tf.Variable(tf.truncated_normal([16], mean=0, stddev=0.01)),
        'conv2w':tf.Variable(tf.truncated_normal([3,3,16,32], mean=0,stddev=0.01)),
        'conv2b':tf.Variable(tf.truncated_normal([32], mean=0, stddev=0.01)),
        'conv3w':tf.Variable(tf.truncated_normal([3,3,32,64], mean=0, stddev=0.01)),
        'conv3b':tf.Variable(tf.truncated_normal([64], mean=0, stddev=0.01)),
        'conv4w':tf.Variable(tf.truncated_normal([3,3,64,128], mean=0, stddev=0.01)),
        'conv4b':tf.Variable(tf.truncated_normal([128], mean=0, stddev=0.01)),
        'conv5w':tf.Variable(tf.truncated_normal([3,3,128,256], mean=0, stddev=0.01)),
        'conv5b':tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.01)),
        'conv6w':tf.Variable(tf.truncated_normal([3,3,256,512], mean=0, stddev=0.01)),
        'conv6b':tf.Variable(tf.truncated_normal([512], mean=0, stddev=0.01)),
        'conv7w':tf.Variable(tf.truncated_normal([3,3,512,1024], mean=0, stddev=0.01)),
        'conv7b':tf.Variable(tf.truncated_normal([1024], mean=0, stddev=0.01)),
        'conv8w':tf.Variable(tf.truncated_normal([3,3,1024,1024], mean=0, stddev=0.01)),
        'conv8b':tf.Variable(tf.truncated_normal([1024], mean=0, stddev=0.01)),
        'conv9w':tf.Variable(tf.truncated_normal([3,3,1024,1024], mean=0, stddev=0.01)),
        'conv9b':tf.Variable(tf.truncated_normal([1024], mean=0, stddev=0.01)),
        'fc10w':tf.Variable(tf.truncated_normal([50176,256], mean=0, stddev=0.01)),
        'fc10b':tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.01)),
        'fc11w':tf.Variable(tf.truncated_normal([256,4096], mean=0, stddev=0.01)),
        'fc11b':tf.Variable(tf.truncated_normal([4096], mean=0, stddev=0.01)),
        'fc12w':tf.Variable(tf.truncated_normal([4096,1470], mean=0, stddev=0.01)),
        'fc12b':tf.Variable(tf.truncated_normal([1470], mean=0, stddev=0.01))
        }

x = tf.placeholder(tf.float32,(None,448,448,3))
keep_prob = tf.placeholder(tf.float32)
yolo_ds = nf.yolo_ds(x,ds_yolo,keep_prob)

file_handler = open('test.txt', 'w')

#fpath = os.path.join(img_root,val_list[0]+'.jpg')
fpath = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages/2008_000002.jpg'
inputs = ut.vocimg_preprocess(fpath)

with tf.Session() as sess2:
    
    sess2.run(tf.global_variables_initializer())  
    resaver = tf.train.import_meta_graph(graph_model)
    resaver.restore(sess2, tf.train.latest_checkpoint(checkpoint_dir))

    for var in ds_yolo:
        sess2.run(ds_yolo[var].assign(tf.get_collection(var)[0]))
        

    idx = 1
    #for fname in val_list:
        
    
    
    print("Test File:{}/{}".format(idx,len(val_list)))
    idx = idx + 1
    prob_label = sess2.run(yolo_ds,feed_dict={x:inputs, keep_prob:0.5})
    results = ut.interpret_output(prob_label[0])
    print(results)
    
#    for i in range(len(results)):
#        
#        cx = int(results[i][1])
#        cy = int(results[i][2])
#        w = int(results[i][3])//2
#        h = int(results[i][4])//2
#        
#        xmin = cx-w
#        ymin = cy-h
#        xmax = cx+w
#        ymax = cy+h 
#        tbb = [xmin, ymin, xmax, ymax]
#        res = ut.eval_by_img(val_list[0], tbb, 0.5)
#        print(res)
#        
#        #file_handler.write("{} {} {} {} {} {}\n".format(fname, results[i][5],xmin,ymin,xmax,ymax))
#        
#    ut.show_results(inputs[0],results)
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        