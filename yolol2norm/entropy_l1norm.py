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


def recursive_create_var(scope, Nlayers, reduce_percent, init_layers):
    
    scope_dict = {}
    
    for idx in range(Nlayers):
        
        scope_name = scope + '_' + str(idx)
        with tf.variable_scope(scope_name):
            
            name_dict = []
            for i in range(len(init_layers)):

                shape = init_layers[i][1]
                
                if idx >= 0: 
                                       
                    if i > 0 and len(shape)  > 1 : shape[-2] = int(init_layers[i-1][1][-1])
                    
                    shape[-1] = int(init_layers[i][1][-1] - reduce_percent*(init_layers[i][1][-1]))
                init_layers[i][1] = shape
                
                if init_layers[i][0] == 'fc12w':  init_layers[i][1][-1] = 1470
                if init_layers[i][0] == 'fc12b':  init_layers[i][1][-1] = 1470          
                if init_layers[i][0] == 'fc10w' and idx == 0:
                    init_layers[i][1][-2] = 40131
                   
          
                tf.get_variable(init_layers[i][0],init_layers[i][1], initializer=tf.contrib.layers.xavier_initializer())

                name_dict.append(init_layers[i][0])
        scope_dict[scope_name] = name_dict 

    return scope_dict


def get_graph_var():

    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    
    
    
def getlearningrate(Epoch):
    
    lr = 1e-3
    
    if Epoch < 10:
        lr = 1e-3 + (1e-2-1e-3)*Epoch/10
    elif Epoch >= 10 and Epoch < 30:
        lr = 1e-2
    elif Epoch >= 30 and Epoch < 60:
        lr = 1e-3
    elif Epoch >= 60:
        lr = 1e-4
        
    return lr
    
datapath = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
if os.path.isdir(datapath):
    batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
    test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
else:
    batch_file = "/home/dashmoment/workspace/dataset/VOCdevkit/VOC2012/JPEGImages"
    test_file = batch_file
    
filename = "../../model/yolol2sum_epoch_SGD/fcann_v1.ckpt"
logfile = '../../log/yolol2sum_epoch_SGD'
checkpoint_dir = '../../model/yolol2sum_epoch_SGD'

train_type = "RMS"
continue_training = 0
epoch_num = 0
Nepoch = 200
batch_size = 4
save_epoch = 20
test_epoch = 50
weight_decay = 0.0005


yolo = YOLO_tiny_tf.YOLO_TF()

with tf.device('/gpu:0'):
    
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    x = tf.placeholder(tf.float32,(None,448,448,3), name='input_batch')
    label = tf.placeholder(tf.float32,(None,1470), name='labels')
    tlabel = tf.placeholder(tf.float32,(None,1470), name='test_labels')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        

    model_ticket = {'root':'yolo_tiny', 'branch':'vanilla'}
    init_layers = mu.model_zoo(model_ticket)
    var_dict = recursive_create_var('recursive', 1, 0.2, init_layers)
    var_list = var_dict['recursive_0']

    with tf.name_scope('Weight_sum'):
        with tf.variable_scope('recursive_0') as scope:
            scope.reuse_variables()
            weight_sum = tf.reduce_sum([0.5*tf.reduce_sum(tf.square(tf.get_variable(x)*weight_decay)) for x in var_list])
            w1 = tf.get_variable(var_list[0])

    lossTicket = {'loss':'g_smoothL1'}
    glosso_train = nf.glosso_train("recursive_0", 'train', x, var_dict, keep_prob, True)
    loss_pair = {'prob':glosso_train, 'gloss':weight_sum}

    class_pre = tf.slice(glosso_train, [0,0],[-1, 980])
    rest_pre = tf.slice(glosso_train, [0, 0],[-1, 980])

#    class_pre = tf.unstack(glosso_train, 490, axis=1)

    loss = mu.loss_zoo(lossTicket, loss_pair, label)
    L2_Solver = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=0.9).minimize(loss)


    tlossTicket = {'loss':'g_smoothL1'}
    glosso_test = nf.glosso_train("recursive_0", 'test', x, var_dict, keep_prob, False)
    tloss_pair = {'prob':glosso_test, 'gloss':weight_sum}
    tloss = mu.loss_zoo(tlossTicket, tloss_pair, tlabel)  


tf.summary.scalar('Train_l2sum',loss,collections=['train'])
tf.summary.scalar('Test_l2sum',tloss,collections=['test'])

merged_summary_train = tf.summary.merge_all('train')
merged_summary_test= tf.summary.merge_all('test')


#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#config = config

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
        

#    for epoch in range(epoch_num,Nepoch):
#
#      print("Start Epoch:{}".format(epoch))
#
#      shufflelist = []
#      lr = getlearningrate(epoch)
#      
#
#      for i in range(0,len(os.listdir(batch_file))//batch_size):
#          
#        summary_idx = len(os.listdir(batch_file))//batch_size*epoch + i
#        print("Epoch:{}, Iteration:{}".format(epoch, summary_idx))
#
#        index = i*batch_size
#        shufflelist, data_labels = mu.gnerate_dl_pairs_voc(yolo, index, batch_file, shufflelist, batch_size, (448,448,3))
#
#        savemodel = False       
#        #print("Learning Rate:{}".format(lr))
#        feeddict = {x:data_labels['data'], label:data_labels['label'], keep_prob:0.5, learning_rate:lr}
#   
#        if summary_idx%save_epoch == 0: savemodel = True
#        mu.train_op(sess, train_type, L2_Solver, loss, feeddict, savemodel, saver, filename, summary, summary_idx)
#        
#        if summary_idx%test_epoch == 0:
#
#            tdata_labels = mu.gnerate_dl_pairs(yolo, test_file, batch_size, (448,448,3))
#            tfeeddict = {x:tdata_labels['data'], tlabel:tdata_labels['label'], keep_prob:0.5}
#            mu.test_op(sess, train_type, tloss, tfeeddict, summary, summary_idx)

#******************Test Code **********************
#    data_labels = mu.gnerate_dl_pairs(yolo, batch_file, batch_size, (448,448,3))
    shufflelist = []
    shufflelist, data_labels = mu.gnerate_dl_pairs_voc(yolo, 0, batch_file, shufflelist, batch_size, (448,448,3))
    feeddict = {x:data_labels['data'], label:data_labels['label'], keep_prob:1, learning_rate:1e-4}    
    
    we1  = sess.run(w1)
    wd1 = data_labels
    w0 = sess.run(glosso_train, feeddict)
    we2 = sess.run(w1)
    wd2 = data_labels
    w  = sess.run(class_pre, feeddict)
#    ww  = sess.run(rest_pre, feeddict)
   
    
    summary_writer.close()
    sess.close()  
    



















