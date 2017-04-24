import tensorflow as tf
import numpy as np
import random_batch as rb

def GAN_vanilla(d_real_logit,d_fake_logit,d_real_prob,d_fake_prob):

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logit, labels=tf.ones_like(d_real_logit)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logit, labels=tf.zeros_like(d_fake_logit)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logit, labels=tf.ones_like(d_fake_logit)))

    return d_loss, g_loss

def GAN_LS(d_real_logit,d_fake_logit,d_real_prob,d_fake_prob):

    d_loss = 0.5 * (tf.reduce_mean((d_real_logit - 1)**2) + tf.reduce_mean(d_fake_logit**2))
    g_loss = 0.5 * tf.reduce_mean((d_fake_logit - 1)**2)

    return d_loss, g_loss

def model_zoo(model_ticket):

    var_list = []

    if model_ticket['root'] == 'yolo_tiny':
        if model_ticket['branch'] =='vanilla':
            var_list = [
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

        if model_ticket['branch'] =='double_cut89':
            var_list = [
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
            ['fc10w',[25088,256]],
            ['fc10b',[256]],
            ['fc11w',[256,4096]],
            ['fc11b',[4096]],
            ['fc12w',[4096,1470]],
            ['fc12b',[1470]]]

    if model_ticket['root'] == 'discriminator':
         if model_ticket['branch'] =='4layer':

            var_list =[ 
            ['fc1w',[1470,1470]],
            ['fc1b',[1470]],
            ['fc2w',[1470,512]],
            ['fc2b',[512]],
            ['fc3w',[512,250]],
            ['fc3b',[250]],
            ['fc4w',[250,1]],
            ['fc4b',[1]]]

    assert len(var_list) > 0, 'Invalid ticket, please check your ticket'

    return var_list

def loss_zoo(ticket, output, label):

    assert 'loss' in ticket, 'Sorry, wrong ticket'

    if ticket['loss'] == 'gL2norm':
        assert 'gloss' in output, 'label and output should contain prob and gloss'
        res_value = tf.subtract(output['prob'], label)
        mloss = tf.sqrt(tf.reduce_sum(tf.square(res_value))) + 0.001*output['gloss']
        gloss = output['gloss']
        loss = [mloss, gloss]

    if ticket['loss'] == 'L2norm':
        res_value = tf.subtract(output, label)
        loss = tf.sqrt(tf.reduce_sum(tf.square(res_value)))

    if ticket['loss'] == 'lsGAN':

        assert 'logit' in label, 'label and output should contain prob and logit'
        loss  = GAN_LS(label['logit'], output['logit'], label['prob'],output['prob'])
    
    if ticket['loss'] == 'vanGAN': #Vanilla GAN

        assert 'logit' in label, 'label and output should contain prob and logit'
        loss  = GAN_vanilla(label['logit'], output['logit'], label['prob'],output['prob'])

    return loss


def create_var_xavier(scope, var_list):

    var_dict = {}
    with tf.variable_scope(scope):
        for n in var_list:
            var_dict[n[0]] = tf.get_variable(n[0], shape=n[1], initializer=tf.contrib.layers.xavier_initializer()) 
            

    return var_dict

def create_var_tnorm(scope, var_list, mean = 0.0, stddev = 0.01):

    var_dict = {}
    with tf.variable_scope(scope):
        for n in var_list:
           
            var_dict[n[0]] = tf.Variable(tf.truncated_normal(n[1], mean=mean, stddev=stddev),name =n[0])

    return var_dict


def train_op(sess, train_type, solver, loss ,feeddict , savemodel, model_saver, model_file, summary, epoch):
     
    if train_type == 'RMS':
        
        sess.run(solver,feed_dict=feeddict)
           
        if savemodel == True:
             
            model_saver.save(sess, model_file, global_step=epoch)
            train_loss = sess.run(loss, feed_dict=feeddict)         
            print("Train Loss: {}".format(train_loss))
           
            if 'train' in summary:#
                sumtrain = sess.run(summary['train'],feed_dict=feeddict) 
                summary['writer'].add_summary(sumtrain, epoch)
    
    if train_type == 'LSGAN':

        assert 'D' in solver and 'G' in solver, "Please define D and G solver by solver['D'] and solver['G']"

        sess.run(solver['D'],feed_dict=feeddict)
        sess.run(solver['G'],feed_dict=feeddict)
        
        if savemodel == True:

            model_saver.save(sess, model_file, global_step=epoch)    
            train_loss = sess.run(loss, feed_dict=feeddict)               
            print("Train Loss D: {}".format(train_loss['D']))
            print("Train Loss G: {}".format(train_loss['G']))

            if 'train' in summary:
                sumtrain = sess.run(summary['train'],feed_dict=feeddict) 
                summary['writer'].add_summary(sumtrain, epoch)

def test_op(sess, train_type ,loss,feeddict,  summary, epoch):
    
    if train_type == 'RMS':             

        test_loss = sess.run(loss, feed_dict=feeddict)
        print("Test Loss: {}".format(test_loss))

    if train_type == 'LSGAN':

        test_loss = sess.run(loss, feed_dict=feeddict)
        print("Test Loss D: {}".format(test_loss['D']))
        print("Test Loss G: {}".format(test_loss['G']))
        
    if 'test'  in summary:
        sumtest = sess.run(summary['test'], feed_dict=feeddict)  
        summary['writer'].add_summary(sumtest, epoch)  


def gnerate_dl_pairs(label_gen, batch_file, batch_size, src_shape = (448,448,3)):

    image_src =  rb.yolo_image_random_batch(batch_file, batch_size, src_shape, np.float32)
    prob_label = label_gen.sess.run(label_gen.fc_19, feed_dict={label_gen.x:image_src})
    dl_pairs = {"data": image_src, "label": prob_label}

    return dl_pairs
    
def quickSummary(key_data):
  
    idx = 0

    print(key_data['collection'][0])

    for k in key_data:       
        if(k != 'collection'):
            
            if len(key_data['collection']) > 1:
                print(key_data['collection'][idx])
                tf.summary.scalar(k,key_data[k],collections=[key_data['collection'][idx]])
                idx = idx + 1
            else:
                tf.summary.scalar(k,key_data[k],collections=[key_data['collection'][0]])


def quickSummary2(key_data):
    print(key_data)
    tf.summary.scalar('Train_RMSE',key_data['Train_RMSE'],collections='train')
    tf.summary.scalar('Group_loss',key_data['Group_loss'],collections='train')
    tf.summary.scalar('Test_RMSE',key_data['Test_RMSE'],collections='test')
    tf.summary.scalar('Test_Group_loss',key_data['Test_Group_loss'],collections='test')

   

