import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2
import model_utility as mut
import utility as ut
import time


tmodel_var_list = [
            ['conv1w',[3,3,3,16]],
            ['conv1b',[16]],
            ['conv2w',[3,3,16,16]],
            ['conv2b',[16]],
            ['conv3w',[3,3,16,32]],
            ['conv3b',[32]],
            ['conv4w',[3,3,32,64]],
            ['conv4b',[64]],
            ['conv5w',[3,3,64,128]],
            ['conv5b',[128]],
            ['conv6w',[3,3,128,256]],
            ['conv6b',[256]],
            ['conv7w',[3,3,256,512]],
            ['conv7b',[512]],
            ['conv8w',[3,3,512,512]],
            ['conv8b',[512]],
            ['conv9w',[3,3,512,512]],
            ['conv9b',[512]],
            ['fc10w',[25088,128]],
            ['fc10b',[128]],
            ['fc11w',[128,2048]],
            ['fc11b',[2048]],
            ['fc12w',[2048,1470]],
            ['fc12b',[1470]]]


discriminator_var = [
            
            ['fc1w',[1470,1470]],
            ['fc1b',[1470]],
            ['fc2w',[1470,512]],
            ['fc2b',[512]],
            ['fc3w',[512,250]],
            ['fc3b',[250]],
            ['fc4w',[250,1]],
            ['fc4b',[1]]]

#Vanilla Yolo

batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
filename = "../../model/yolo_lsgan/fcann_v1.ckpt"
logfile = '../../log/yolo_lsgan'
graph_model = '../../model/yolo_lsgan/fcann_v1.ckpt-0.meta'
checkpoint_dir = '../../model/yolo_lsgan'

continue_training = 1
loop_num = 5500
d_loop_num = 3
batch_size = 64

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,(None,448,448,3))
label = tf.placeholder(tf.float32,(None,1470))

yolo = YOLO_tiny_tf.YOLO_TF()
ds_yolo = mut.create_var_xavier('train',tmodel_var_list)
dis_var = mut.create_var_xavier('discriminator', discriminator_var)

theta_D = []
theta_G = []

for i in tmodel_var_list:
    theta_G.append(ds_yolo[i[0]])
for i in discriminator_var:
    theta_D.append(dis_var[i[0]])


##Train Phase
yolo_ds_train = nf.yolo_ds_all("yolo_train", x,ds_yolo,keep_prob, True)
d_real_logit, d_real_prob = nf.discriminator('discriminator', label,dis_var)
d_fake_logit, d_fake_prob = nf.discriminator('discriminator', yolo_ds_train,dis_var)

d_loss, g_loss = ut.GAN_LS("gan_train", d_real_logit, d_fake_logit, d_real_prob,d_fake_prob)

tf.summary.scalar("G_loss",g_loss,collections=['train'])
tf.summary.scalar("D_loss",d_loss,collections=['train'])

D_solver = tf.train.AdamOptimizer(1e-4).minimize(d_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(1e-4).minimize(g_loss, var_list=theta_G)
res_value = tf.subtract(yolo_ds_train, label)
loss = tf.sqrt(tf.reduce_sum(tf.square(res_value)))
tf.summary.scalar("Train_RMSE",loss, collections=['train'])


##Test Phase
yolo_ds_test = nf.yolo_ds_all("yolo_test", x,ds_yolo,keep_prob, False)
td_real_logit, td_real_prob = nf.discriminator('discriminator', label,dis_var)
td_fake_logit, td_fake_prob = nf.discriminator('discriminator', yolo_ds_test,dis_var)
td_loss, tg_loss = ut.GAN_LS("gan_test",td_real_logit, td_fake_logit,  td_real_prob,td_fake_prob)

tres_value = tf.subtract(yolo_ds_test, label)
tloss = tf.sqrt(tf.reduce_sum(tf.square(tres_value)))

tf.summary.scalar("Real_RMSE",tloss, collections=['test'])
tf.summary.scalar("tG_loss",tg_loss,collections=['test'])
tf.summary.scalar("tD_loss",td_loss,collections=['test'])

merged_summary_train = tf.summary.merge_all('train')
merged_summary_test= tf.summary.merge_all('test')


with tf.Session() as sess:
    
    summary_writer = tf.summary.FileWriter(logfile, sess.graph)  
    saver = tf.train.Saver()    
   
    if continue_training !=0:

        resaver = tf.train.Saver()
        resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
        
    else:
        sess.run(tf.global_variables_initializer())  
        
    image_src =  rb.yolo_image_random_batch(batch_file, batch_size, (448,448,3), np.float32)
    prob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_src})
        
#    c1 = sess.run(ds_yolo["conv2w"])

        
    for i in range(loop_num, 1000000):
       
       print("Epoch:{}".format(i))
       

       for j in range(d_loop_num):
            print("D Epoch:{}".format(j))
            image_src =  rb.yolo_image_random_batch(batch_file, batch_size, (448,448,3), np.float32)
            prob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_src})
            sess.run(D_solver,feed_dict={x:image_src, label:prob_label, keep_prob:0.5})

       image_src =  rb.yolo_image_random_batch(batch_file, batch_size, (448,448,3), np.float32)
       prob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_src})
       #sess.run(D_solver,feed_dict={x:image_src, label:prob_label, keep_prob:0.5})
       sess.run(G_solver,feed_dict={x:image_src, label:prob_label, keep_prob:0.5})
                          
       if i%200 == 0:
        
            dloss, gloss,sumtrain = sess.run([d_loss, g_loss,merged_summary_train], feed_dict={x:image_src, label:prob_label, keep_prob:0.5})         
            print("Train Loss D: {}".format(dloss))
            print("Train Loss G: {}".format(gloss))
            summary_writer.add_summary(sumtrain, i)
            saved_model = saver.save(sess, filename, global_step=i)
           
       if i%500 == 0:
           image_test = rb.yolo_image_random_batch(test_file, batch_size, (448,448,3), np.float32)
           tprob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_test})
           tdloss, tgloss, sumtest = sess.run([td_loss, tg_loss,merged_summary_test], feed_dict={x:image_test, label:tprob_label, keep_prob:0.5})
           print("Test Loss D: {}".format(tdloss))
           print("Test Loss G: {}".format(tgloss))
           summary_writer.add_summary(sumtest, i)

    summary_writer.close()
    sess.close()  
    
    
##    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    