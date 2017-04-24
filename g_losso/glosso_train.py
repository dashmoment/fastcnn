import  tensorflow as tf
import numpy as np
import sys
sys.path.append('/home/ubuntu/workspace/fastcnn/src/yolo')

import model_utility as mu
import yolo_netfactory as nf
import YOLO_tiny_tf

batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
filename = "../../model/glosso/fcann_v1.ckpt"
logfile = '../../log/glosso'
graph_model = '../../model/glosso/fcann_v1.ckpt-0.meta'
checkpoint_dir = '../../model/glosso'

train_type = "RMS"
continue_training = 0
loop_num = 0
batch_size = 64
save_epoch = 10
test_epoch = 10


yolo = YOLO_tiny_tf.YOLO_TF()
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,(None,448,448,3))
label = tf.placeholder(tf.float32,(None,1470))
tlabel = tf.placeholder(tf.float32,(None,1470))


def recursive_create_var(scope, Nlayers, reduce_percent, init_layers):
    
    scope_dict = {}
    
    for idx in range(Nlayers):
        
        scope_name = scope + '_' + str(idx)
        with tf.variable_scope(scope_name):
            
            name_dict = []
            for i in range(len(init_layers)):

                shape = init_layers[i][1]
                
                if idx != 0: 
                                       
                    if i > 0 and len(shape)  > 1 : shape[-2] = int(init_layers[i-1][1][-1])
                    
                    shape[-1] = int(init_layers[i][1][-1] - reduce_percent*(init_layers[i][1][-1]))
                init_layers[i][1] = shape
                
                if init_layers[i][0] == 'fc12w':  init_layers[i][1][-1] = 1470
                if init_layers[i][0] == 'fc12b':  init_layers[i][1][-1] = 1470          
                if init_layers[i][0] == 'fc10w' and idx == 1:
                    init_layers[i][1][-2] = 40131
                   
          
                tf.get_variable(init_layers[i][0],init_layers[i][1], initializer=tf.contrib.layers.xavier_initializer())

                name_dict.append(init_layers[i][0])
        scope_dict[scope_name] = name_dict 

    return scope_dict


def get_graph_var():

    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    

model_ticket = {'root':'yolo_tiny', 'branch':'vanilla'}
init_layers = mu.model_zoo(model_ticket)
var_dict = recursive_create_var('recursive', 1, 0.2, init_layers)
var_list = var_dict['recursive_0']

var = {}
sum_stack = []
  
with tf.variable_scope('recursive_0') as scope:
    scope.reuse_variables()
    for index in range(0,len(var_list),2):
        var[var_list[index]] = tf.get_variable(var_list[index])
        var[var_list[index+1]] = tf.get_variable(var_list[index+1])
        
        if index < 18:
            wsum = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(var[var_list[index]]), axis=[0,1,2])))
            bsum = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(var[var_list[index+1]]))))
            
        else:
            wsum = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(var[var_list[index]]), axis=0)))
            bsum = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(var[var_list[index+1]]))))
           
        sum_stack.append(wsum)
        sum_stack.append(bsum)
            
        
    glosso = tf.reduce_sum(tf.stack(sum_stack,0))        


lossTicket = {'loss':'gL2norm'}
glosso_train = nf.glosso_train("recursive_0", 'train', x, var_dict, keep_prob, True)

loss = {'prob':glosso_train, 'gloss':glosso}
gloss = mu.loss_zoo(lossTicket, loss, label)
L2_Solver = tf.train.AdamOptimizer(1e-4).minimize(gloss[0])


#tlossTicket = {'loss':'gL2norm'}
#glosso_test = nf.glosso_train("recursive_0", 'test', x, var_dict, keep_prob, False)
#tloss = {'prob':glosso_test, 'gloss':glosso}
#tgloss = mu.loss_zoo(tlossTicket, tloss, tlabel)  


tf.summary.scalar('Train_RMSE',gloss[0],collections=['train'])
#tf.summary.scalar('Train_Glossp',gloss[1],collections=['train'])
#tf.summary.scalar('Test_RMSE',tgloss[0],collections=['test'])
#tf.summary.scalar('Test_Glosso',tgloss[1],collections=['test'])

merged_summary_train = tf.summary.merge_all('train')
#merged_summary_test= tf.summary.merge_all('test')



with tf.Session() as sess:
    
    summary_writer = tf.summary.FileWriter(logfile, sess.graph)  
    summary = {'writer':summary_writer, 'train':merged_summary_train}
    #summary = {'writer':summary_writer, 'train':merged_summary_train, 'test':merged_summary_test} 
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
        
       data_labels = mu.gnerate_dl_pairs(yolo, batch_file, batch_size, (448,448,3))
       feeddict = {x:data_labels['data'], label:data_labels['label'], keep_prob:0.5}
 
       if i%save_epoch == 0: savemodel = True
       mu.train_op(sess, train_type, L2_Solver, gloss[0], feeddict, savemodel, saver, filename, summary, i)
      
#       if i%test_epoch == 0:#

#           tdata_labels = mu.gnerate_dl_pairs(yolo, test_file, batch_size, (448,448,3))
#           tfeeddict = {x:tdata_labels['data'], tlabel:tdata_labels['label'], keep_prob:0.5}
#           mu.test_op(sess, train_type, tgloss[0], tfeeddict, summary, i)

    
#get_graph_var()


















