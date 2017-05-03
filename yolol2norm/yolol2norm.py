import  tensorflow as tf
import sys
import os
sys.path.append('/home/ubuntu/workspace/fastcnn/src/yolo')

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


batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
filename = "../../model/test/fcann_v1.ckpt"
logfile = '../../log/test'
checkpoint_dir = '../../model/test'

train_type = "RMS"
continue_training = 0
epoch_num = 0
Nepoch = 200
loop_num = 0
batch_size = 64
save_epoch = 20
test_epoch = 50

yolo = YOLO_tiny_tf.YOLO_TF()
keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
x = tf.placeholder(tf.float32,(None,448,448,3), name='input_batch')
label = tf.placeholder(tf.float32,(None,1470), name='labels')
tlabel = tf.placeholder(tf.float32,(None,1470), name='test_labels')
learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    

model_ticket = {'root':'yolo_tiny', 'branch':'vanilla'}
init_layers = mu.model_zoo(model_ticket)
var_dict = recursive_create_var('recursive', 1, 0.2, init_layers)
var_list = var_dict['recursive_0']


lossTicket = {'loss':'yoloL2norm'}
glosso_train = nf.glosso_train("recursive_0", 'train', x, var_dict, keep_prob, True)
loss_pair = {'prob':glosso_train}
loss = mu.loss_zoo(lossTicket, loss_pair, label)
L2_Solver = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)


tlossTicket = {'loss':'yoloL2norm'}
glosso_test = nf.glosso_train("recursive_0", 'test', x, var_dict, keep_prob, False)
tloss_pair = {'prob':glosso_test}
tloss = mu.loss_zoo(tlossTicket, tloss_pair, tlabel)  


tf.summary.scalar('Train_l2sum',loss,collections=['train'])
tf.summary.scalar('Test_l2sum',tloss,collections=['test'])

merged_summary_train = tf.summary.merge_all('train')
merged_summary_test= tf.summary.merge_all('test')



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
        
#    data_labels = mu.gnerate_dl_pairs(yolo, batch_file, batch_size, (448,448,3))
#    feeddict = {x:data_labels['data'], label:data_labels['label'], keep_prob:0.5, learning_rate:1e-4}
##    l = sess.run(var_list[0], feed_dict = feeddict)
#    
#    with tf.variable_scope("recursive_0") as scope:
#        scope.reuse_variables()
#        ds_yolo = {}
#        for i in range(len(var_list)):
#            ds_yolo[var_list[i]] = tf.get_variable(var_list[i])
#            
#    c = sess.run(ds_yolo['conv2w'])

    for epoch in range(epoch_num,Nepoch):

      print("Start Epoch:{}".format(epoch))

      shufflelist = []
      
      print(len(os.listdir(batch_file))//batch_size)

      for i in range(0,len(os.listdir(batch_file))//batch_size):
          
        summary_idx = len(os.listdir(batch_file))//batch_size*epoch + i

        print("Epoch:{}, Iteration:{}".format(epoch, i))

        index = i*batch_size
        print(index)
        shufflelist, data_labels = mu.gnerate_dl_pairs_voc(yolo, index, batch_file, shufflelist, batch_size, (448,448,3))

        savemodel = False       
        feeddict = {x:data_labels['data'], label:data_labels['label'], keep_prob:0.5, learning_rate:1e-6}
   
        if summary_idx%save_epoch == 0: savemodel = True
        mu.train_op(sess, train_type, L2_Solver, loss, feeddict, savemodel, saver, filename, summary, summary_idx)
        
        if summary_idx%test_epoch == 0:

            tdata_labels = mu.gnerate_dl_pairs(yolo, test_file, batch_size, (448,448,3))
            tfeeddict = {x:tdata_labels['data'], tlabel:tdata_labels['label'], keep_prob:0.5}
            mu.test_op(sess, train_type, tloss, tfeeddict, summary, summary_idx)

    summary_writer.close()
    sess.close()  
    



















