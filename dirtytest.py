import tensorflow as tf
import operator
import numpy as np

tensor_1 = tf.Variable(tf.random_normal([3,3,20]))
reduce_t1 = tf.reduce_mean(tensor_1, axis=[0,1]) 
tensor_2 = tf.Variable(tf.random_normal([3,3,3]))
intra_lose = [tf.Variable(tf.zeros([1])), tf.Variable(tf.zeros([1])), tf.Variable(tf.zeros([1])), tf.Variable(tf.zeros([1])), tf.Variable(tf.zeros([1]))]
loss = tf.Variable(tf.zeros([1]))

Nfilter = 3


var_list = {
        'conv1': [3,3,20],
        'conv2': [3,3,10],
        
        }


def recursive_create_var(scope, Nlayers, reduce_percent, init_layers):
    
    v1 = []
    scope_dict = {}
    
    for idx in range(Nlayers):
        
        scope_name = scope + '_' + str(idx)
        with tf.variable_scope(scope_name):
            
            name_dict = []
            for i in init_layers:

            	shape = init_layers[i]
            	if(idx != 0): shape[2] = int(init_layers[i][2] - reduce_percent*(init_layers[i][2]))
            	init_layers[i] = shape
            	print("shape:{}".format(init_layers[i]))
                
                if scope_name == "tesnor_0" and i == "conv1":
                    v1 = tf.get_variable(i,init_layers[i], initializer=tf.contrib.layers.xavier_initializer())
                else:
                    tf.get_variable(i,init_layers[i], initializer=tf.contrib.layers.xavier_initializer())

                name_dict.append(i)
        scope_dict[scope_name] = name_dict 
    return scope_dict,v1
             
        

def partition_by_rank(sess, inputs, output, ptype = 'Mean'):
    
    inputs = sess.run(inputs)
    output_dim = sess.run(output)
    
    partition_shape = np.zeros(inputs.shape)
    
    if ptype == 'Mean':
        
        mean = sess.run(tf.reduce_mean(inputs, axis=[0,1])) 
        idx = range(len(mean))
        mean_dict = dict(zip(idx, abs(mean)))
        sorted_var = sorted(mean_dict.items(), key=operator.itemgetter(1))

        reduce_dim = output_dim.shape
        
        for i in range(reduce_dim[2]):
            partition_shape[:,:,sorted_var[i][0]] = np.ones([3,3])
        
        pres = sess.run(tf.dynamic_partition(inputs, partition_shape, 2))
       
        
        pres = pres[1].reshape(output_dim.shape)
        out = sess.run(tf.assign(output,pres))
        
        return out,sorted_var
        

def get_graph_var():

	[n.name for n in tf.get_default_graph().as_graph_def().node]
	print(n)



with tf.Session() as sess:
    
    d, v = recursive_create_var("tesnor", 5, 0.1,var_list)
    mean_true = 0
    
    idx = 0
    
    
    for k in d:
        with tf.variable_scope(k) as scope:
            scope.reuse_variables()
            for v in d[k]:
               intra_lose[idx] = tf.add(intra_lose[idx],tf.sqrt(tf.reduce_sum(tf.square(tf.get_variable(v)))))
            idx = idx + 1
        loss = tf.add( loss, intra_lose[idx-1])
            
            
    sess.run(tf.global_variables_initializer())
    mean = sess.run(intra_lose[0])
    mean2 = sess.run(intra_lose[1])
    loss2 = sess.run(loss)
    
    
    
    with tf.variable_scope("tesnor_1") as scope:
        scope.reuse_variables()
        for v in d["tesnor_1"]:
           mean_true = mean_true + sess.run(tf.sqrt(tf.reduce_sum(tf.square(tf.get_variable(v)))))
    
            
       
    
    
#    v2 = sess.run(v)
    

#    with tf.variable_scope("tesnor_0") as scope:
#        scope.reuse_variables()
#        v1 = tf.get_variable("conv1", [3,3,20])
#        v2 = tf.get_variable("conv2", [3,3,10])
#        test = sess.run(v1)
#        test2 = sess.run(v1)
#
#    with tf.variable_scope("tesnor_1") as scope:
#        scope.reuse_variables()
#        v3 = tf.get_variable("conv1", [3,3,18])
#        
#       
#
#        out , sort = partition_by_rank(sess, v1, v3)
#        test3 = sess.run(v3)
#    print(np.array_equal(test, test2))
#
#    get_graph_var()
#    var1 = sess.run(tensor_1)
#    meanvar1 = sess.run(reduce_t1)
#    
#    idx = range(len(meanvar1))
#    mean_dict = dict(zip(idx, abs(meanvar1)))
#    sorted_x = sorted(mean_dict.items(), key=operator.itemgetter(1))
#    
#    
    
#    for i in range(len(sorted_x) - Nfilter):
#        
#        partition_shape[:,:,sorted_x[i][0]] = np.ones([3,3])
#    
#    pres = sess.run(tf.dynamic_partition(tensor_1, partition_shape, 2))
#    res = pres[1].reshape([3,3,10])
#    
#    var2 = sess.run(tf.assign(tensor_2,res))
#    
#    
#    test = var1[:,:,3]
#    sum_t = 0
#    
#    for i in range(3):
#        for j in range(3):
#            sum_t = sum_t + test[i][j]
#            
#    mean_t = sum_t/9