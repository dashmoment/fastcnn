import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2
import model_utility as mut
import time


def init_yolo_weight(sess,yolo_cls, ds_yolo):
    
    key_pairs = {
            'conv1w':'Variable',
            'conv1b':'Variable_1',
            'conv2w':'Variable_2',
            'conv2b':'Variable_3',
            'conv3w':'Variable_4',
            'conv3b':'Variable_5',
            'conv4w':'Variable_6',
            'conv4b':'Variable_7',
            'conv5w':'Variable_8',
            'conv5b':'Variable_9',
            'conv6w':'Variable_10',
            'conv6b':'Variable_11',
            'conv7w':'Variable_12',
            'conv7b':'Variable_13',
            'conv8w':'Variable_14',
            'conv8b':'Variable_15',
            'conv9w':'Variable_16',
            'conv9b':'Variable_17',
            'fc10w':'Variable_18',
            'fc10b':'Variable_19',
            'fc11w':'Variable_20',
            'fc11b':'Variable_21',
            'fc12w':'Variable_22',
            'fc12b':'Variable_23',
            
            }
    
    for var in key_pairs:
        yolo_var = yolo.sess.run(tf.get_default_graph().get_tensor_by_name(key_pairs[var]+':0'))
        op = tf.assign(ds_yolo[var], yolo_var)
        sess.run(op)

tmodel_var_list = [
            ['conv1w',[3,3,3,16]],
            ['conv1b',[16]],
            ['conv2w',[3,3,16,32]],
            ['conv2b',[32]],
            ['conv3w',[3,3,32,64]],
            ['conv3b',[64]],
            ['conv4w',[3,3,64,128]],
            ['conv4b',[128]],
            ['conv5w',[3,3,128,256]],
            ['conv5b',[256]],
            ['conv6w',[3,3,256,512]],
            ['conv6b',[512]],
            ['conv7w',[3,3,512,1024]],
            ['conv7b',[1024]],
            ['conv8w',[3,3,1024,1024]],
            ['conv8b',[1024]],
            ['conv9w',[3,3,1024,1024]],
            ['conv9b',[1024]],
            ['fc10w',[50176,256]],
            ['fc10b',[256]],
            ['fc11w',[256,4096]],
            ['fc11b',[4096]],
            ['fc12w',[4096,1470]],
            ['fc12b',[1470]]]

#Vanilla Yolo
scope = 'train'
yolo = YOLO_tiny_tf.YOLO_TF()
ds_yolo = mut.create_var_tnorm(scope,tmodel_var_list)


#for var in ds_yolo:
#    tf.summary.histogram(var, ds_yolo[var], collections=['train'])

batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
filename = "../../model/test/fcann_v1.ckpt"
logfile = '../../log/test'
graph_model = '../../model/test/fcann_v1.ckpt-4000.meta'
checkpoint_dir = '../../model/test'

continue_training = 1
loop_num = 5000
batch_size = 64

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,(None,448,448,3))
label = tf.placeholder(tf.float32,(None,1470))#


##Train Phase
yolo_ds_train = nf.yolo_ds_train("yolo_train", x,ds_yolo,keep_prob)
res_value = tf.subtract(yolo_ds_train, label)
loss = tf.sqrt(tf.reduce_sum(tf.square(res_value)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
tf.summary.scalar("train_RMSE",loss, collections=['train'])#

##Test Phase
tres_value = tf.subtract(yolo_ds_train, label)
tloss = tf.sqrt(tf.reduce_sum(tf.square(tres_value)))
tf.summary.scalar("test_RMSE",tloss, collections=['test'])#

merged_summary_train = tf.summary.merge_all('train')
merged_summary_test= tf.summary.merge_all('test')

with tf.Session() as sess2:
    
    summary_writer = tf.summary.FileWriter(logfile, sess2.graph)  
    saver = tf.train.Saver()    
    
 
    if continue_training !=0:
        
        resaver = tf.train.import_meta_graph(graph_model)
        resaver.restore(sess2, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
        
    else:
        sess2.run(tf.global_variables_initializer())  
        init_yolo_weight(sess2,yolo,ds_yolo)
    
#    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    for i in range(loop_num, 100000000):
        
        print("Epoch:{}".format(i))
        image_src =  rb.yolo_image_random_batch(batch_file, batch_size, (448,448,3), np.float32)
        prob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_src})
        #sess2.run(train_step,feed_dict={x:image_src, label:prob_label, keep_prob:0.5})
                           
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
    
        c1 = sess2.run(ds_yolo["conv1w"])
    
    summary_writer.close()
    sess2.close()  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    