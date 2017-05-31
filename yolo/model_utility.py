import tensorflow as tf
import numpy as np
import random_batch as rb
import math

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

    #============Group Losso==============
    if ticket['loss'] == 'gL2norm':
        assert 'gloss' in output, 'label and output should contain prob and gloss'
        res_value = tf.subtract(output['prob'], label)
        mloss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(res_value), axis=1))) + 0.01*output['gloss']
        gloss = output['gloss']
        loss = [mloss, gloss]

    if ticket['loss'] == 'gRMSE':
        assert 'gloss' in output, 'label and output should contain prob and gloss'
        rmse = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(output['prob'], label)), axis=1)))
        mloss = rmse + 0.01*output['gloss']
        gloss = output['gloss']
        loss = [mloss, gloss]

    if ticket['loss'] == 'gkl_divergence':
        assert 'gloss' in output, 'label and output should contain prob and gloss'

        prob_logit = tf.nn.softmax(output['prob'])
        prob_label = tf.nn.softmax(label)
        mloss =  tf.reduce_mean(tf.reduce_sum(prob_logit*tf.log(prob_logit/prob_label),axis=1))
        gloss = output['gloss']
        loss = [mloss, gloss]


    if ticket['loss'] == 'g_smoothL1':
        assert 'gloss' in output, 'label and output should contain prob and gloss'

        gloss = output['gloss']
        res_value = tf.abs(tf.subtract(output['prob'],label))
        
        smoothL1 = tf.cast(tf.less(res_value,1), tf.float32)
        invsmoothL1 = tf.cast(tf.less(smoothL1,0.5),tf.float32)

        r1 = tf.multiply(tf.square(res_value), smoothL1)*0.5
        r2 = tf.multiply((res_value - 0.5), invsmoothL1)       
        res = tf.add(r1,r2)

        loss = tf.add(tf.reduce_mean(tf.reduce_sum(res, axis=1)),gloss)

    if ticket['loss'] == 'yolo_kl_l1':
        assert 'gloss' in output, 'label and output should contain prob and gloss'
        
        predict = output['prob']
        gloss = output['gloss']
        
        label_cls = tf.slice(label, [0,0],[-1, 980])
        #label_conf = tf.slice(label, [0,980],[-1, 98])
        label_offset = tf.slice(label, [0, 980],[-1, 490])
    
        pre_cls = tf.slice(predict, [0,0],[-1, 980])
        #pre_conf = tf.slice(gloss, [0,980],[-1, 98])
        pre_offset = tf.slice(predict, [0, 980],[-1, 490])

        #===========KL Divergence of each class====================
        label_cls = tf.nn.softmax(tf.reshape(label_cls, [-1,49,20] ))
        pre_cls = tf.nn.softmax(tf.reshape(pre_cls, [-1,49,20]))
        kl_loss = tf.reduce_sum(tf.multiply(pre_cls, tf.log(tf.divide(pre_cls,label_cls))), axis=[1,2])

        #============SmoothL1 for rest parts======================
        res_value = tf.abs(tf.subtract(pre_offset,label_offset))
        smoothL1 = tf.cast(tf.less(res_value,1), tf.float32)
        invsmoothL1 = tf.cast(tf.less(smoothL1,0.5),tf.float32)
        r1 = tf.multiply(tf.square(res_value), smoothL1)*0.5
        r2 = tf.multiply((res_value - 0.5), invsmoothL1)       
        offset_loss = tf.reduce_sum(tf.add(r1,r2), axis=1)#

        loss = tf.add(tf.reduce_mean(tf.add(kl_loss, offset_loss)),gloss)
        

    #============General Loss==============
    if ticket['loss'] == 'L2norm':
        res_value = tf.subtract(output['prob'], label)
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(res_value), axis=1)))
        
    if ticket['loss'] == 'RMSE':
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(output['prob'], label)), axis=1)))  
       
    if ticket['loss'] == 'L1norm':
        res_value = tf.abs(tf.subtract(output['prob'],label))
        loss = tf.reduce_mean(tf.reduce_sum(res_value, axis=1))
    
    if ticket['loss'] == 'smoothL1':
        
        res_value = tf.abs(tf.subtract(output['prob'],label))
        smoothL1 = tf.cast(tf.less(res_value,1), tf.float32)
        invsmoothL1 = tf.cast(tf.less(smoothL1,0.5),tf.float32)

        r1 = tf.multiply(tf.square(res_value), smoothL1)*0.5
        r2 = tf.multiply((res_value - 0.5), invsmoothL1)       
        res = tf.add(r1,r2)
        
        loss = tf.reduce_mean(tf.reduce_sum(res, axis=1))
             
    #=============GAN===============
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


def gnerate_dl_pairs_voc(label_gen, index, batchpath, shufflelist, batchsize, imagesize=(448,448,3)):

    shufflelist, image_src =  rb.voc_image_random_batch(index, batchpath, shufflelist, batchsize, imagesize)
    prob_label = label_gen.sess.run(label_gen.fc_19, feed_dict={label_gen.x:image_src})
    dl_pairs = {"data": image_src, "label": prob_label}

    return shufflelist, dl_pairs
    
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

   
def weight_pruning(yolo_obj, sess, var_dict, var_scope):
    
    for var in var_dict[var_scope]:
        tensors = yolo_obj.sess.run(tf.get_default_graph().get_tensor_by_name(key_pairs[var]+':0'))
        
        with tf.variable_scope(var_scope) as scope:
            scope.reuse_variables() 
            tensors_s =  sess.run(tf.get_variable(var))
              
        if np.shape(tensors) != np.shape(tensors_s):
                        
                if (len(np.shape(tensors))) > 1:
                
                    if (len(np.shape(tensors))) > 2:                         
                        dim_sum = np.sum(np.sum(np.sum(tensors, axis=0),axis=0), axis=1)
                    else:
                        dim_sum = np.sum(tensors, axis=1)
                    
                    dim_sum = np.abs(dim_sum)
                    dim_sort = np.unravel_index(dim_sum.argsort(axis=None), dims=int(len(dim_sum)))
                    del_list = [dim_sort[0][x] for x in range(np.shape(tensors)[-2] -  np.shape(tensors_s)[-2])]
                    
                    dim_array = np.delete(tensors,del_list, -2)
                    
                    if (len(np.shape(tensors))) > 2:        
                        kernel_sum = np.sum(np.sum(np.sum(dim_array,axis=0), axis=0),axis=0)
                    else:
                        kernel_sum = np.sum(dim_array, axis=0)
                    
                    kernel_sum = np.abs(kernel_sum)
                    kernel_sort = np.unravel_index(kernel_sum.argsort(axis=None), dims=int(len(kernel_sum)))
                    kdel_list = [kernel_sort[0][x] for x in range(np.shape(tensors)[-1] -  np.shape(tensors_s)[-1])]
                    kernel_array = np.delete(dim_array, kdel_list, -1)
                
                else:
                    kernel_array = np.delete(tensors, kdel_list,0)
                
        else:
            kernel_array = tensors

    
        with tf.variable_scope(var_scope) as scope:
                scope.reuse_variables() 
                op = tf.assign(tf.get_variable(var), kernel_array)
                sess.run(op)
                tensors_s =  sess.run(tf.get_variable(var))
                
                
                
def weight_pruning_ind(yolo_obj, sess, var_dict, var_scope):
    
    for var in var_dict[var_scope]:
         tensors = yolo_obj.sess.run(tf.get_default_graph().get_tensor_by_name(key_pairs[var]+':0'))
    
         with tf.variable_scope(var_scope) as scope:
            scope.reuse_variables() 
            tensors_s =  sess.run(tf.get_variable(var))
              
         if np.shape(tensors) != np.shape(tensors_s):
         
             
             if (len(np.shape(tensors))) > 1:
            
                if (len(np.shape(tensors))) > 2:     

                    dim_array = np.zeros((np.shape(tensors_s)[0],np.shape(tensors_s)[1],np.shape(tensors_s)[2],np.shape(tensors)[3]), dtype=np.float32)                    
                    dim_sum = np.sum(np.sum(tensors, axis=0),axis=0)
                    dim_sum = np.abs(dim_sum)
                    
                    del_axis = []
                    
                    for i in range(dim_sum.shape[1]):
                        axis_sum = dim_sum[:,i]
                        axis_sort = np.unravel_index(axis_sum.argsort(axis=None), dims=int(len(axis_sum)))
                        del_axis = [axis_sort[0][x] for x in range(np.shape(tensors)[-2] -  np.shape(tensors_s)[-2])]
                        tmp = tensors[:,:,:,i]
                        stmp = np.delete(tmp,del_axis, -1)
                        dim_array[:,:,:,i] = stmp
                        
                        
                else:  
                     
                     dim_sum = np.sum(tensors, axis=1)
                     dim_sum = np.abs(dim_sum)
                     dim_sort = np.unravel_index(dim_sum.argsort(axis=None), dims=int(len(dim_sum)))
                     del_list = [dim_sort[0][x] for x in range(np.shape(tensors)[-2] -  np.shape(tensors_s)[-2])]
                     dim_array = np.delete(tensors,del_list, -2)
                     
                     
                if (len(np.shape(tensors))) > 2:        
                        kernel_sum = np.sum(np.sum(np.sum(dim_array,axis=0), axis=0),axis=0)
                else:
                        kernel_sum = np.sum(dim_array, axis=0)
                    
                kernel_sum = np.abs(kernel_sum)
                kernel_sort = np.unravel_index(kernel_sum.argsort(axis=None), dims=int(len(kernel_sum)))
                kdel_list = [kernel_sort[0][x] for x in range(np.shape(tensors)[-1] -  np.shape(tensors_s)[-1])]
                kernel_array = np.delete(dim_array, kdel_list, -1)
                
             else:
                kernel_array = np.delete(tensors, kdel_list,0)
                
       
         else:   
             kernel_array = tensors

    
         with tf.variable_scope(var_scope) as scope:
             scope.reuse_variables() 
             op = tf.assign(tf.get_variable(var), kernel_array)
             sess.run(op)
             tensors_s = sess.run(tf.get_variable(var)) 



