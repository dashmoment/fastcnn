import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2
import utility as ut

import time



#Vanilla Yolo

yolo = YOLO_tiny_tf.YOLO_TF()

with tf.name_scope("new"):
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


batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
filename = "../../model/yolo_ds/fcann_v1.ckpt"
logfile = '../../log/yolo_ds'
graph_model = '../../model/yolo_ds/fcann_v1.ckpt-62000.meta'
checkpoint_dir = '../../model/yolo_ds'


batch_size = 1

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,(None,448,448,3))
label = tf.placeholder(tf.float32,(None,1470))


yolo_ds = nf.yolo_ds(x,ds_yolo,keep_prob)
res_value = tf.subtract(yolo_ds, label)
loss = tf.sqrt(tf.reduce_sum(tf.square(res_value)))


fromfile = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages/2008_000002.jpg'
w,h,inputs = ut.vocimg_preprocess(fromfile)




with tf.Session() as sess:
    
    
    sess.run(tf.global_variables_initializer())  
        
    resaver = tf.train.import_meta_graph(graph_model)
    resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    
    for var in ds_yolo:
        sess.run(ds_yolo[var].assign(tf.get_collection(var)[0]))
    a = sess.run(ds_yolo[var])
    #image_src =  rb.yolo_image_random_batch(batch_file, batch_size, (448,448,3), np.float32)
    
    s = time.clock()
    prob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:inputs})
    e = time.clock()
    print("Origin Elapse:{}".format(e-s))
    
    s = time.clock()
    prob_label_new = sess.run(yolo_ds,feed_dict={x:inputs, keep_prob:0.5})
    e = time.clock()
    print("DS Elapse:{}".format(e-s))
    
    
    loss = sess.run(loss,feed_dict={x:inputs, label:prob_label, keep_prob:0.5})
    print(loss)
                      
    #results_o = ut.interpret_output(prob_label[0])
#    yolo.show_results(img_resized,results)
    
    
    results_new = ut.interpret_output(prob_label_new[0],w,h)
#    yolo.show_results(img_resized,results_new)
    print(results_new)
    
    
   
    sess.close()  
    


 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    