import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2
import model_utility as mut
import utility as ut
import time

batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
filename = "../../model/yolo_dk/fcann_v1.ckpt"
logfile = '../../log/yolo_dk'
checkpoint_dir = '../../model/yolo_dk'

train_type = "RMS"
continue_training = 1
loop_num = 23643
d_loop_num = 3
batch_size = 64
save_epoch = 200
test_epoch = 500

modelTicket_G = {'root':'yolo_tiny', 'branch':'double_cut89'}
modelTicket_D = {'root':'discriminator', 'branch':'4layer'}


keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,(None,448,448,3))
test = tf.placeholder(tf.float32,(None,448,448,3))
label = tf.placeholder(tf.float32,(None,1470))

yolo = YOLO_tiny_tf.YOLO_TF()


gen_var = mut.create_var_xavier('train',mut.model_zoo(modelTicket_G))
dis_var = mut.create_var_xavier('discriminator', mut.model_zoo(modelTicket_D))

theta_D = []
theta_G = []

for i in mut.model_zoo(modelTicket_G):
    theta_G.append(gen_var[i[0]])
for i in mut.model_zoo(modelTicket_D):
    theta_D.append(dis_var[i[0]])


##Train Phase
yolo_ds_train = nf.yolo_dinception("yolo_train", x, gen_var, keep_prob, True)
lossTicket = {'loss':'L2norm'}
loss = mut.loss_zoo(lossTicket, yolo_ds_train, label)
L2_Solver = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=theta_G)

#Test Phase
yolo_ds_test = nf.yolo_dinception("yolo_test", x ,gen_var,keep_prob, False)
tloss = mut.loss_zoo(lossTicket, yolo_ds_test, label)


train_summary = {'Train_RMSE': loss, 'Test_RMSE':tloss, 'collection':['train','test']}
mut.quickSummary(train_summary)
merged_summary_train = tf.summary.merge_all('train')
merged_summary_test= tf.summary.merge_all('test')

#------LSGAN---------
#d_real_logit, d_real_prob = nf.discriminator('discriminator', label,dis_var)
#d_fake_logit, d_fake_prob = nf.discriminator('discriminator', yolo_ds_train,dis_var)#
#d_loss, g_loss = ut.GAN_LS("gan_train", d_real_logit, d_fake_logit, d_real_prob,d_fake_prob)#
#D_solver = tf.train.AdamOptimizer(1e-4).minimize(d_loss, var_list=theta_D)
#G_solver = tf.train.AdamOptimizer(1e-4).minimize(g_loss, var_list=theta_G)


with tf.Session() as sess:
    
    summary_writer = tf.summary.FileWriter(logfile, sess.graph)  
    summary = {'writer':summary_writer, 'train':merged_summary_train, 'test':merged_summary_test} 
    saver = tf.train.Saver()    
    
    if continue_training !=0:

        resaver = tf.train.Saver()
        resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
        
    else:
        sess.run(tf.global_variables_initializer())  
    
    
    for i in range(loop_num, 1000000):
       
       print("Epoch:{}".format(i))
       savemodel = False
       
       data_labels = mut.gnerate_dl_pairs(yolo, batch_file, batch_size, (448,448,3))
       feeddict = {x:data_labels['data'], label:data_labels['label'], keep_prob:0.5}

       
       if i%save_epoch == 0: savemodel = True
       mut.train_op(sess, train_type, L2_Solver, loss, feeddict, savemodel,saver, filename, summary, i)
       
       if i%test_epoch == 0:

           tdata_labels = mut.gnerate_dl_pairs(yolo, test_file, batch_size, (448,448,3))
           tfeeddict = {x:tdata_labels['data'], label:tdata_labels['label'], keep_prob:0.5}
           mut.test_op(sess, train_type, tloss, tfeeddict, summary, i)

    #--------------------Test Function-------------------    
#    image_src =  rb.yolo_image_random_batch(batch_file, batch_size, (448,448,3), np.float32)
#    
#    elapse = 0
#    
#    for i in range(1000):
#        s = time.clock()
#        prob_label = sess.run(yolo_ds_test, feed_dict={x:image_src})
##        prob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_src})
#        e = time.clock()
#        print("elapse:{}".format(e-s))
#        elapse = elapse + e - s
#    print("Elapse:{}".format(elapse/1000))
#    c1 = sess.run(ds_yolo["conv2w"])
#--------------------End of Test Function-------------------    

    summary_writer.close()
    sess.close()  
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    