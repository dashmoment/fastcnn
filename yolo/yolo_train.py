import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2
import model_utility as mut
import time



#Vanilla Yolo
scope = 'train'
yolo = YOLO_tiny_tf.YOLO_TF()

#Target Model
model_ticket = {'root':yolo_tiny, 'branch':'double_cut89'}
ds_yolo = mut.create_var_tnorm(scope,mut.model_zoo(model_ticket))


batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
filename = "../../model/test/fcann_v1.ckpt"
logfile = '../../log/test'
graph_model = '../../model/test/fcann_v1.ckpt-4000.meta'
checkpoint_dir = '../../model/test'

continue_training = 1
loop_num = 5900
batch_size = 64

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,(None,448,448,3))
label = tf.placeholder(tf.float32,(None,1470))#


##Train Phase
yolo_ds_train = nf.yolo_ds_all("yolo_train", x,ds_yolo,keep_prob, True)
res_value = tf.subtract(yolo_ds_train, label)
loss = tf.sqrt(tf.reduce_sum(tf.square(res_value)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
tf.summary.scalar("train_RMSE",loss, collections=['train'])#

##Test Phase
yolo_ds_test = nf.yolo_ds_all("yolo_train", x,ds_yolo,keep_prob, False)
tres_value = tf.subtract(yolo_ds_test, label)
tloss = tf.sqrt(tf.reduce_sum(tf.square(tres_value)))
tf.summary.scalar("test_RMSE",tloss, collections=['test'])

merged_summary_train = tf.summary.merge_all('train')
merged_summary_test= tf.summary.merge_all('test')

with tf.Session() as sess2:
    
    summary_writer = tf.summary.FileWriter(logfile, sess2.graph)  
    saver = tf.train.Saver()    
    
 
    if continue_training !=0:
        
        
        saver.restore(sess2, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
        
    else:
        sess2.run(tf.global_variables_initializer())  
    
#    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    for i in range(loop_num, 100000000):
        
        
        print("Epoch:{}".format(i))
        image_src =  rb.yolo_image_random_batch(batch_file, batch_size, (448,448,3), np.float32)
        prob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_src})
        sess2.run(train_step,feed_dict={x:image_src, label:prob_label, keep_prob:0.5})
                           
        if i%500 == 0:
         
            train_loss, sumtrain = sess2.run([loss,merged_summary_train], feed_dict={x:image_src, label:prob_label, keep_prob:0.5})         
            print("Train Loss: {}".format(train_loss))
            summary_writer.add_summary(sumtrain, i)
            
            saved_model = saver.save(sess2, filename, global_step=i)
            
            if i%1000 == 0:
                image_test = rb.yolo_image_random_batch(test_file, batch_size, (448,448,3), np.float32)
                prob_label2 = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_test})
                test_loss, sumtest = sess2.run([tloss,merged_summary_test], feed_dict={x:image_test, label:prob_label2, keep_prob:0.5})
                print("Test Loss: {}".format(test_loss))
                summary_writer.add_summary(sumtest, i)
    
        #c1 = sess2.run(ds_yolo["conv2w"])
    
    summary_writer.close()
    sess2.close()  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    