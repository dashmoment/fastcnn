import tensorflow as tf
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy import misc
import purgeinvalid_img as pi
import net_factory
import time



def randombatch(batchfile):
    filenames = tf.train.match_filenames_once(batchfile)
    file_queue =  tf.train.string_input_producer( filenames,  shuffle=True)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(file_queue)
    image_orig = tf.image.decode_jpeg(image_file)
    image = tf.image.resize_images(image_orig, [227, 227])
    image.set_shape((227, 227, 3))   
    num_preprocess_threads = 5
    min_queue_examples = 500
    batch_size = 128
    
    images = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + (num_preprocess_threads +50)*batch_size,
            min_after_dequeue=min_queue_examples)
    
    return images

#pi.purgeinvalidandRGB_img("../../dataset/VOC2012/JPEGImages") #Cleanup Invalid file in directory
x = tf.placeholder(tf.float32,(None,227,227,3))
label = tf.placeholder(tf.float32,(None,1000))
image_submean = tf.subtract(x, tf.reduce_mean(x))
keep_prob = tf.placeholder(tf.float32)



with tf.name_scope("Vanilla"):
    fcw , fcb, loss_van = net_factory.vanilla_alex_full(image_submean)
    tf.summary.histogram('fc8W', fcw, collections=['train'])
    tf.summary.histogram('fc8b', fcb, collections=['train'])

with tf.name_scope("Mini"):

        with tf.name_scope("conv1"):
            s_h = 8; s_w = 8
            rconv1W = tf.Variable(tf.random_normal([11,11,3,96],stddev=0.01))
            rconv1b = tf.Variable(tf.random_normal([96],mean= 0,stddev= 0.01)) 
            conv1_in = tf.nn.conv2d(image_submean, rconv1W, strides=[1,s_h,s_w,1], padding='VALID')
            conv1_add = tf.nn.bias_add(conv1_in, rconv1b)
            conv1 = tf.nn.relu(conv1_add)
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
            
        #conv2
        with tf.name_scope("conv2"):
            k_h = 5; k_w = 5; c_o = 256; s_h = 2; s_w = 2
            rconv2W = tf.Variable(tf.random_normal([k_h,k_w,96,256],stddev=0.01))
            rconv2b = tf.Variable(tf.random_normal([256],mean= 0,stddev= 0.01))        
            conv2_in = tf.nn.conv2d(lrn1, rconv2W, strides=[1,s_h,s_w,1], padding='VALID')
            conv2_add = tf.nn.bias_add(conv2_in, rconv2b)
            conv2 = tf.nn.relu(conv2_add)
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)

        #conv3
        with tf.name_scope("conv3"):
        #conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1
            rconv3W = tf.Variable(tf.random_normal([k_h,k_w,256,c_o],stddev=0.01))
            rconv3b = tf.Variable(tf.random_normal([384],mean= 0,stddev= 0.01))
            conv3_in = tf.nn.conv2d(lrn2, rconv3W, strides=[1,s_h,s_w,1], padding='SAME')
            conv3_add = tf.nn.bias_add(conv3_in, rconv3b)
            conv3 = tf.nn.relu(conv3_add)
        
        #conv4
        with tf.name_scope("conv4"):
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1
            rconv4W = tf.Variable(tf.random_normal([k_h,k_w,384,c_o],stddev=0.01))
            rconv4b = tf.Variable(tf.random_normal([384],mean= 0,stddev= 0.01))
            conv4_in = tf.nn.conv2d(conv3, rconv4W, strides=[1,s_h,s_w,1], padding='SAME')
            conv4_add = tf.nn.bias_add(conv4_in, rconv4b)
            conv4 = tf.nn.relu(conv4_add)
        
        
        #conv5
        with tf.name_scope("conv5"):
            k_h = 3; k_w = 3; c_o = 256; s_h = 2; s_w = 2
            rconv5W = tf.Variable(tf.random_normal([k_h,k_w,384,c_o],stddev=0.01))
            rconv5b = tf.Variable(tf.random_normal([c_o],mean= 0,stddev= 0.01))
            conv5_in = tf.nn.conv2d(conv4, rconv5W, strides=[1,s_h,s_w,1], padding='VALID')
            conv5_add = tf.nn.bias_add(conv5_in, rconv5b)
            conv5 = tf.nn.relu(conv5_add)


        #fc6
        #fc(4096, name='fc6')
        with tf.name_scope("fc6"):
            len_input = int(np.prod(conv5.get_shape()[1:]))
            
            rfc6W = tf.Variable(tf.random_normal([len_input, 4096], stddev=0.01))
            rfc6b = tf.Variable(tf.random_normal([4096], mean= 0,stddev= 0.01))
            fc6 = tf.nn.relu_layer(tf.reshape(conv5, [-1, len_input]), rfc6W, rfc6b)

        with tf.name_scope("dropout1"):
            dropout1 = tf.nn.dropout(fc6, keep_prob)
        
        #fc7
        #fc(4096, name='fc7')
        with tf.name_scope("fc7"):
            rfc7W = tf.Variable(tf.random_normal([4096, 4096], stddev=0.01))
            rfc7b = tf.Variable(tf.random_normal([4096], mean= 0,stddev= 0.01))
            fc7 = tf.nn.relu_layer(dropout1, rfc7W, rfc7b)

        with tf.name_scope("dropout2"):
            dropout2 = tf.nn.dropout(fc7, keep_prob)
        
        #fc8
        #fc(1000, relu=False, name='fc8')
        with tf.name_scope("fc8"):
            rfc8W = tf.Variable(tf.random_normal([4096, 1000], stddev=0.01))
            rfc8b = tf.Variable(tf.random_normal([1000], mean= 0,stddev= 0.01))
            fc8 = tf.nn.xw_plus_b(dropout2, rfc8W, rfc8b)     
            prob = tf.nn.softmax(fc8)
            #prob = fc8

        tf.summary.histogram('conv1W', rconv1W, collections=['train'])
        tf.summary.histogram('conv1b', rconv1b , collections=['train'])
        tf.summary.histogram('conv2W', rconv2W, collections=['train'])
        tf.summary.histogram('conv2b', rconv2b, collections=['train'])
        tf.summary.histogram('conv3W', rconv3W, collections=['train'])
        tf.summary.histogram('conv3b', rconv3b, collections=['train'])
        tf.summary.histogram('conv4W', rconv4W, collections=['train'])
        tf.summary.histogram('conv4b', rconv4b, collections=['train'])
        tf.summary.histogram('conv5W', rconv5W, collections=['train'])
        tf.summary.histogram('conv5b', rconv5b, collections=['train'])
        tf.summary.histogram('r_fc8w', rfc8W, collections=['train'])
        tf.summary.histogram('r_fc8b', rfc8b, collections=['train'])
    

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc8))
#test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc8))
#tf.summary.histogram("label",label)

'''
filename = "../model/test/fcann_v1.ckpt"
logfile = '../log/test'
graph_model = '../model/test/fcann_v1.ckpt-2.meta'
checkpoint_dir = '../model/test'
'''

batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train/*.jpg"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val/*.jpg"
filename = "../model/voc2012_reg_do/fcann_v1.ckpt"
logfile = '../log/voc2012_reg_do'
graph_model = '../model/voc2012_reg_do/fcann_v1.ckpt-4000.meta'
checkpoint_dir = '../model/voc2012_reg_do'

continue_training = 0
loop_num = 0
_keep_prob = 0.5
    
sample_batch = randombatch(batch_file)
test_batch = randombatch(test_file)

res_value = tf.subtract(prob, label)
loss = tf.sqrt(tf.reduce_sum(tf.square(res_value)))
tf.summary.scalar("cross_entrpy",loss, collections=['train'])

tres_value = tf.subtract(prob, label)
test_loss =  tf.sqrt(tf.reduce_sum(tf.square(tres_value)))
test_cross_entrpy = tf.summary.scalar("test_cross_entrpy",test_loss, collections=['test'])

merged_summary_op = tf.summary.merge_all('train')
merged_summary_op2 = tf.summary.merge_all('test')

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    
    summary_writer = tf.summary.FileWriter(logfile, sess.graph)  
    saver = tf.train.Saver()  
    sess.run(tf.global_variables_initializer())  

    if continue_training !=0:
        resaver = tf.train.import_meta_graph(graph_model)
        #resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir)) 
        resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))          
        sess.run(rconv1W.assign(tf.get_collection('w1')[0]))
        sess.run(rconv1b.assign(tf.get_collection('b1')[0]))
        sess.run(rconv2W.assign(tf.get_collection('w2')[0]))
        sess.run(rconv2b.assign(tf.get_collection('b2')[0]))
        sess.run(rconv3W.assign(tf.get_collection('w3')[0]))
        sess.run(rconv3b.assign(tf.get_collection('b3')[0]))
        sess.run(rconv4W.assign(tf.get_collection('w4')[0]))
        sess.run(rconv4b.assign(tf.get_collection('b4')[0]))
        sess.run(rconv5W.assign(tf.get_collection('w5')[0]))
        sess.run(rconv5b.assign(tf.get_collection('b5')[0]))
        sess.run(rfc6W.assign(tf.get_collection('fc6W')[0]))
        sess.run(rfc6b.assign(tf.get_collection('fc6b')[0]))
        sess.run(rfc7W.assign(tf.get_collection('fc7W')[0]))
        sess.run(rfc7b.assign(tf.get_collection('fc7b')[0]))
        sess.run(rfc8W.assign(tf.get_collection('fc8W')[0]))
        sess.run(rfc8b.assign(tf.get_collection('fc8b')[0]))
        continue_training = 0
    
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord) 
    
    try:
        #while not coord.should_stop():
        for i in range(loop_num, 20000000):

            
            print("Training Step: {}".format(i))
            
            image_tensor = sess.run(sample_batch)
            label_van = sess.run(loss_van, feed_dict = {x:image_tensor})
                   
#            for input_im_ind in range(label_van.shape[0]):
#                inds = np.argsort(label_van)[input_im_ind,:]#

#                for j in range(0,len(inds)):
#                    if j < len(inds)-1 : 
#                        label_van[input_im_ind, inds[j]] = 0
#                    if j >= len(inds) - 1:
#                        label_van[input_im_ind, inds[j]] = 1
#                assert label_van[input_im_ind, inds[-2]] == 0
#                assert label_van[input_im_ind, inds[-1]] == 1
               
            sess.run(train_step, feed_dict = {x:image_tensor, label:label_van, keep_prob:_keep_prob}) 

            if i%10 == 0:
                
                cost, summary = sess.run([loss,merged_summary_op], feed_dict = {x:image_tensor, label:label_van, keep_prob:_keep_prob}) 
                isummary = sess.run(tf.summary.image("batch{}".format(i), sample_batch, max_outputs=3))
                #grad_vals = sess.run([grad[0] for grad in grads])
                summary_writer.add_summary(summary, i)
                summary_writer.add_summary(isummary, i)
                
                tf.add_to_collection("w1", rconv1W)
                tf.add_to_collection("b1", rconv1b)
                tf.add_to_collection("w2", rconv2W)
                tf.add_to_collection("b2", rconv2b)
                tf.add_to_collection("w3", rconv3W)
                tf.add_to_collection("b3", rconv3b)
                tf.add_to_collection("w4", rconv4W)
                tf.add_to_collection("b4", rconv4b)
                tf.add_to_collection("w5", rconv5W)
                tf.add_to_collection("b5", rconv5b)
                tf.add_to_collection("fc6W", rfc6W)
                tf.add_to_collection("fc6b", rfc6b)
                tf.add_to_collection("fc7W", rfc7W)
                tf.add_to_collection("fc7b", rfc7b)
                tf.add_to_collection("fc8W", rfc8W)
                tf.add_to_collection("fc8b", rfc8b)
                
                print("total loss:{}".format(cost))
                print("Save Model:{}".format(filename))
                saved_model = saver.save(sess, filename, global_step=i)
            

                if i%10 == 0:
                 
                    test_tensor = sess.run(test_batch)
                    label_van = sess.run(loss_van, feed_dict = {x:test_tensor})
    #                for input_im_ind in range(label_van.shape[0]):
    #                    inds = np.argsort(label_van)[input_im_ind,:]#

    #                    for j in range(0,len(inds)):
    #                        if j < len(inds)-1 : 
    #                            label_van[input_im_ind, inds[j]] = 0
    #                        if j >= len(inds) - 1:
    #                            label_van[input_im_ind, inds[j]] = 1
    #                    assert label_van[input_im_ind, inds[-2]] == 0
    #                    assert label_van[input_im_ind, inds[-1]] == 1
                   
                    test_cost ,testsummary = sess.run([test_loss, merged_summary_op2], feed_dict = {x:test_tensor, label:label_van, keep_prob:_keep_prob}) 
                    summary_writer.add_summary(testsummary, i)
                    print("test loss:{}".format(test_cost))
           
           
            
            
           
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)
        
    summary_writer.close()
    sess.close()  
   








