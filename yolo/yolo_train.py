import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2

import time

batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
filename = "../../model/yolo/fcann_v1.ckpt"
logfile = '../../log/yolo'
graph_model = '../../model/yolo/fcann_v1.ckpt-10.meta'
checkpoint_dir = '../../model/yolo'

#Vanilla Yolo
yolo = YOLO_tiny_tf.YOLO_TF()

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




for var in ds_yolo:
    tf.summary.histogram(var, ds_yolo[var], collections=['train'])

     
continue_training = 0
loop_num = 0
batch_size = 64

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,(None,448,448,3))
label = tf.placeholder(tf.float32,(None,1470))

#Train Phase
yolo_ds_train = nf.yolo_vanilla_train(x,ds_yolo,keep_prob)
#yolo_ds_train = nf.yolo_ds_train(x,ds_yolo,keep_prob)
res_value = tf.subtract(yolo_ds_train, label)
loss = tf.sqrt(tf.reduce_sum(tf.square(res_value)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
tf.summary.scalar("train_RMSE",loss, collections=['train'])

#Test Phase
yolo_ds = nf.yolo_vanilla(x,ds_yolo,keep_prob)
#yolo_ds = nf.yolo_ds(x,ds_yolo,keep_prob)
tres_value = tf.subtract(yolo_ds, label)
tloss = tf.sqrt(tf.reduce_sum(tf.square(tres_value)))
tf.summary.scalar("test_RMSE",tloss, collections=['test'])

merged_summary_train = tf.summary.merge_all('train')
merged_summary_test= tf.summary.merge_all('test')

with tf.Session() as sess2:
    
    summary_writer = tf.summary.FileWriter(logfile, sess2.graph)  
    saver = tf.train.Saver()    
    sess2.run(tf.global_variables_initializer())  
    
    if continue_training !=0:
        
        resaver = tf.train.import_meta_graph(graph_model)
        resaver.restore(sess2, tf.train.latest_checkpoint(checkpoint_dir))
        
        for var in ds_yolo:
            sess2.run(ds_yolo[var].assign(tf.get_collection(var)[0]))
        
        continue_training = 0
    
    for i in range(loop_num, 200000000):
        
        print("Epoch:{}".format(i))
        image_src =  rb.yolo_image_random_batch(batch_file, batch_size, (448,448,3), np.float32)
        prob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_src})
        sess2.run(train_step, feed_dict={x:image_src, label:prob_label, keep_prob:0.5})
        
               
        if i%500 == 0:
         
            train_loss, sumtrain = sess2.run([loss,merged_summary_train], feed_dict={x:image_src, label:prob_label, keep_prob:0.5})         
            print("Train Loss: {}".format(train_loss))
            summary_writer.add_summary(sumtrain, i)
            
            for var in ds_yolo:          
                tf.add_to_collection(var, ds_yolo[var])
            saved_model = saver.save(sess2, filename, global_step=i)
            
            if i%1000 == 0:
                image_test = rb.yolo_image_random_batch(test_file, batch_size, (448,448,3), np.float32)
                prob_label2 = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_test})
                test_loss, sumtest = sess2.run([tloss,merged_summary_test], feed_dict={x:image_test, label:prob_label2, keep_prob:0.5})
                print("Test Loss: {}".format(test_loss))
                summary_writer.add_summary(sumtest, i)
    
    
    
    summary_writer.close()
    sess2.close()  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    