import  tensorflow as tf
import sys
import os 
libpath = '/home/ubuntu/workspace/fastcnn/src/yolo'

if os.path.isdir(libpath):
    sys.path.append(libpath)
else:
    sys.path.append('/home/dashmoment/workspace/fastcnn/yolo')
    

import model_utility as mu
import yolo_netfactory as nf
import YOLO_tiny_tf
import numpy as np


def recursive_create_var(scope, Nlayers, reduce_percent, init_layers):
    
    scope_dict = {}
    
    for idx in range(Nlayers):
        
        scope_name = scope + '_' + str(idx)
        with tf.variable_scope(scope_name):
            
            name_dict = []
            
            for i in range(len(init_layers)):

                shape = init_layers[i][1]
                
                if idx >= 1: 
                                       
                    if i > 0 and len(shape)  > 1 : shape[-2] = int(init_layers[i-1][1][-1])
                    
                    shape[-1] = int(init_layers[i][1][-1]*reduce_percent)
                    
                    
                init_layers[i][1] = shape
                
                if init_layers[i][0] == 'fc12w':  init_layers[i][1][-1] = 1470
                if init_layers[i][0] == 'fc12b':  init_layers[i][1][-1] = 1470          
                if init_layers[i][0] == 'fc10w':
                
                    if idx == 0:
                        init_layers[i][1][-2] = 50176
                    if idx == 1:
                        init_layers[i][1][-2] = 40131
                    if idx == 2:
                        init_layers[i][1][-2] = 32095
                   
          
                tf.get_variable(init_layers[i][0],init_layers[i][1], initializer=tf.contrib.layers.xavier_initializer())

                name_dict.append(init_layers[i][0])
                
        scope_dict[scope_name] = name_dict 

    return scope_dict


def get_graph_var():

    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    
    
    
def getlearningrate(Epoch, loopnum, totalloop):
    
    lr = 1e-3
    
    if Epoch < 1:
        lr = 1e-3 + (1e-2-1e-3)*loopnum/totalloop
    elif Epoch >= 1 and Epoch < 75:
        lr = 1e-2
    elif Epoch >= 75 and Epoch < 105:
        lr = 1e-3
    elif Epoch >= 105:
        lr = 1e-4
        
    return lr
    

def yolo_loss(scope, predict, tvalid_class, tvalid_xy,tvalid_wh, tlabel_xy, tlabel_wh, tconf_weight, tlabel_conf, batch_size):
    
    with tf.name_scope(scope):
        pre_cls = tf.reshape(tf.slice(predict, [0,0],[-1, 980]), (-1,7,7,20))
        pre_conf = tf.reshape(tf.slice(predict, [0,980],[-1, 98]), (-1,7,7,2))
        pre_offset = tf.reshape(tf.slice(predict, [0,1078],[-1, 392]), (-1,7,7,2,4))
        
        pre_xy, pre_wh = tf.split(pre_offset,2,axis=4)
        
        class_loss = tf.reduce_sum(tf.square(tf.subtract(tf.multiply(pre_cls, tvalid_class), tvalid_class)))
        
        box_xy_loss = tf.reduce_sum(tf.square(tf.subtract(tf.multiply(pre_xy, tvalid_xy),tlabel_xy)))
        pre_wh_sqrt = tf.sqrt(tf.multiply(pre_wh, tvalid_wh)+1e-4)
        label_wh_sqrt = tf.sqrt(tlabel_wh+1e-4)
        box_wh_loss = tf.reduce_sum(tf.square(tf.subtract(pre_wh_sqrt,label_wh_sqrt)))
        tbias_loss = 5*(box_xy_loss  + box_wh_loss)
        
        tconf_loss =  tf.reduce_sum(tf.multiply(tf.square(tf.subtract(tlabel_conf, pre_conf)), tconf_weight))
        tloss = (class_loss + tbias_loss + tconf_loss)/batch_size
    
    return tloss



datapath = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
if os.path.isdir(datapath):
    batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
    test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
else:
    batch_file = "/home/dashmoment/workspace/dataset/VOCdevkit/VOC2012/JPEGImages"
    test_file = batch_file
    
filename = "../../model/yololoss_init_0.8/fcann_v1.ckpt"
checkpoint_dir = '../../model/yololoss_init_0.8'
logfile = '../../log/yololoss_init_0.8'


train_type = "RMS"
continue_training = 0
epoch_num = 0
Nepoch = 200
batch_size = 64
save_epoch = 200
test_epoch = 500
weight_decay = 0.0005


yolo = YOLO_tiny_tf.YOLO_TF()
varscope = 'recursive_1'
shrinkratio = 0.8


with tf.device('/gpu:1'):
    
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    x = tf.placeholder(tf.float32,(None,448,448,3), name='input_batch')
    label = tf.placeholder(tf.float32,(None,1470), name='labels')
    tlabel = tf.placeholder(tf.float32,(None,1470), name='test_labels')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    model_ticket = {'root':'yolo_tiny', 'branch':'vanilla'}
    init_layers = mu.model_zoo(model_ticket)
    var_dict = recursive_create_var('recursive', 2, shrinkratio, init_layers)
    var_list = var_dict[varscope]

    with tf.name_scope('Weight_sum'):
        with tf.variable_scope(varscope) as scope:
            scope.reuse_variables()
            weight_sum = tf.reduce_sum([0.5*tf.reduce_sum(tf.square(tf.get_variable(x)*weight_decay)) for x in var_list])
            
    #============Yolo Loss===============
   
    tvalid_class = tf.placeholder(tf.float32,(None,7,7,20), name='class')
    tvalid_xy = tf.placeholder(tf.float32,(None,7,7,2,2), name='xy')
    tvalid_wh = tf.placeholder(tf.float32,(None,7,7,2,2), name='wh')
    tlabel_xy = tf.placeholder(tf.float32,(None,7,7,2,2), name='label_xy')
    tlabel_wh = tf.placeholder(tf.float32,(None,7,7,2,2), name='label_wh')
    tconf_weight = tf.placeholder(tf.float32,(None,7,7,2), name='label_wh')
    tlabel_conf = tf.placeholder(tf.float32,(None,7,7,2), name='label_wh')
        
    
        
    #==============End of Yolo Loss==================================
    
    glosso_train = nf.glosso_train(varscope, 'train', x, var_dict, keep_prob, True)
    loss = yolo_loss('train_loss',glosso_train, tvalid_class, tvalid_xy, tvalid_wh, tlabel_xy, tlabel_wh, tconf_weight, tlabel_conf, batch_size)
    loss = tf.add(weight_sum, loss)    
    L2_Solver = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=0.9).minimize(loss)
    
    glosso_test = nf.glosso_train(varscope, 'test', x, var_dict, keep_prob, False)   
    tloss = yolo_loss('test_loss', glosso_test, tvalid_class, tvalid_xy, tvalid_wh, tlabel_xy, tlabel_wh, tconf_weight, tlabel_conf, batch_size)
    tloss = tf.add(weight_sum, tloss)

tf.summary.scalar('Train_l2sum',loss,collections=['train'])
tf.summary.scalar('learning_rate',learning_rate,collections=['train'])
tf.summary.scalar('Test_l2sum',tloss,collections=['test'])

merged_summary_train = tf.summary.merge_all('train')
merged_summary_test= tf.summary.merge_all('test')


config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config = config) as sess:
    
    summary_writer = tf.summary.FileWriter(logfile, sess.graph)  
    summary = {'writer':summary_writer, 'train':merged_summary_train, 'test':merged_summary_test} 
    saver = tf.train.Saver()    
    
    if continue_training !=0:

        resaver = tf.train.Saver()
        resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        continue_training = 0
        
    else:
        sess.run(tf.global_variables_initializer())  
        mu.weight_pruning_ind(yolo, sess, var_dict, varscope)

        
    for epoch in range(epoch_num,Nepoch):

      print("Start Epoch:{}".format(epoch))

      shufflelist = []
      
      for i in range(0,len(os.listdir(batch_file))//batch_size):

        lr = getlearningrate(epoch,i,len(os.listdir(batch_file))//batch_size)
          
        summary_idx = len(os.listdir(batch_file))//batch_size*epoch + i
        print("Epoch:{}, Iteration:{}".format(epoch, summary_idx))

        index = i*batch_size
        shufflelist, data_labels = mu.gnerate_dl_pairs_voc(yolo, index, batch_file, shufflelist, batch_size, (448,448,3))

        
        prob_threshold = 0.1
     
     
        rlabel_boxes = np.zeros((7,7,2,4))
        label_probs = np.reshape(data_labels['label'][:,0:980],(-1,7,7,20))
        label_scales = np.reshape(data_labels['label'][:,980:1078],(-1,7,7,2))
        label_boxes = np.reshape(data_labels['label'][:,1078:],(-1,7,7,2,4))
        conf = np.zeros((label_probs.shape[0], 7,7,2,20))
         
        for i in range(2):
               for j in range(20):
                   conf[:,:,:,i,j] = np.multiply(label_probs[:,:,:,j],label_scales[:,:,:,i])
                   filter_mat_probs = np.array(conf>=prob_threshold,dtype='int')  
        
        valid_class = np.multiply(conf, filter_mat_probs)
        valid_class = np.sum(valid_class, axis=-2)
        valid_class = np.array(valid_class>0,dtype='int')  
        
        valid_box = np.multiply(conf, filter_mat_probs)
        valid_box = np.sum(valid_box, axis=-1)
        valid_box = np.array(valid_box>0,dtype='int')     
        
        for i in range(2):
            for j in range(4):
                label_boxes[:,:,:,i,j] = np.multiply(label_boxes[:,:,:,i,j], valid_box[:,:,:,i])
        
        label_xy , label_wh = np.split(label_boxes,2,axis=-1)
        valid_xy = np.array(label_xy>0,dtype='int')  
        valid_wh = np.array(label_wh>0,dtype='int') 
        
        label_conf = np.multiply(valid_box, label_scales)
        w_obj = np.array(np.abs(label_conf)>0,dtype='int')
        w_nobj = np.array(np.abs(label_conf)==0,dtype='int')
        w = w_obj + 0.5*w_nobj
        
        feeddict = {x:data_labels['data'], keep_prob:0.5, learning_rate:lr  , tvalid_class:valid_class,  tvalid_xy:valid_xy, tlabel_xy:label_xy,tvalid_wh:valid_wh, tlabel_wh:label_wh, tlabel_conf:label_conf, tconf_weight:w}

        savemodel = False
        if summary_idx%save_epoch == 0: savemodel = True
        mu.train_op(sess, train_type, L2_Solver, loss, feeddict, savemodel, saver, filename, summary, summary_idx)
        
        if summary_idx%test_epoch == 0:

            tdata_labels = mu.gnerate_dl_pairs(yolo, test_file, batch_size, (448,448,3))
            
            prob_threshold = 0.1
        
            rlabel_boxes = np.zeros((7,7,2,4))
            label_probs = np.reshape(tdata_labels['label'][:,0:980],(-1,7,7,20))
            label_scales = np.reshape(tdata_labels['label'][:,980:1078],(-1,7,7,2))
            label_boxes = np.reshape(tdata_labels['label'][:,1078:],(-1,7,7,2,4))
            conf = np.zeros((label_probs.shape[0], 7,7,2,20))
             
            for i in range(2):
                   for j in range(20):
                       conf[:,:,:,i,j] = np.multiply(label_probs[:,:,:,j],label_scales[:,:,:,i])
                       filter_mat_probs = np.array(conf>=prob_threshold,dtype='int')  
            
            valid_class = np.multiply(conf, filter_mat_probs)
            valid_class = np.sum(valid_class, axis=-2)
            valid_class = np.array(valid_class>0,dtype='int')  
            
            valid_box = np.multiply(conf, filter_mat_probs)
            valid_box = np.sum(valid_box, axis=-1)
            valid_box = np.array(valid_box>0,dtype='int')     
            
            for i in range(2):
                for j in range(4):
                    label_boxes[:,:,:,i,j] = np.multiply(label_boxes[:,:,:,i,j], valid_box[:,:,:,i])
                    
            tfeeddict = {x:tdata_labels['data'], keep_prob:1  , tvalid_class:valid_class,  tvalid_xy:valid_xy, tlabel_xy:label_xy,tvalid_wh:valid_wh, tlabel_wh:label_wh, tlabel_conf:label_conf, tconf_weight:w}
        
            mu.test_op(sess, train_type, tloss, tfeeddict, summary, summary_idx)


    



















