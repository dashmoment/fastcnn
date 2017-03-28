import tensorflow as tf
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy import misc
import purgeinvalid_img as pi
import net_factory
import time



def randombatch():
    filenames = tf.train.match_filenames_once("../../dataset/ilsvrc_train/*.jpg")
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


continue_training = 0
loop_num = 0

pi.purgeinvalidandRGB_img("../../dataset/ilsvrc_train") #Cleanup Invalid file in directory
x = tf.placeholder(tf.float32,(None,227,227,3))

image_submean = tf.subtract(x, tf.reduce_mean(x))
sample_batch = randombatch()

with tf.name_scope("Vanilla"):
    fcw , fcb, loss_van = net_factory.vanilla_alex_full(image_submean)
    tf.summary.histogram('fc8W', fcw)
    tf.summary.histogram('fc8b', fcb)


with tf.name_scope("New_network"):

     
        s_h = 8; s_w = 8
        rconv1W = tf.Variable(tf.random_normal([11,11,3,96],stddev=0.1))
        rconv1b = tf.Variable(tf.random_normal([96],mean= -0.5,stddev= 0.3)) 
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
        k_h = 5; k_w = 5; c_o = 256; s_h = 2; s_w = 2
        rconv2W = tf.Variable(tf.random_normal([k_h,k_w,96,256],stddev=0.1))
        rconv2b = tf.Variable(tf.random_normal([256],mean= -0.5,stddev= 0.3))        
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
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1
        rconv3W = tf.Variable(tf.random_normal([k_h,k_w,256,c_o],stddev=0.1))
        rconv3b = tf.Variable(tf.random_normal([384],mean= -0.5,stddev= 0.3))
        conv3_in = tf.nn.conv2d(lrn2, rconv3W, strides=[1,s_h,s_w,1], padding='SAME')
        conv3_add = tf.nn.bias_add(conv3_in, rconv3b)
        conv3 = tf.nn.relu(conv3_add)
        
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1
        rconv4W = tf.Variable(tf.random_normal([k_h,k_w,384,c_o],stddev=0.1))
        rconv4b = tf.Variable(tf.random_normal([384],mean= -0.5,stddev= 0.3))
        conv4_in = tf.nn.conv2d(conv3, rconv4W, strides=[1,s_h,s_w,1], padding='SAME')
        conv4_add = tf.nn.bias_add(conv4_in, rconv4b)
        conv4 = tf.nn.relu(conv4_add)
        
        
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 2; s_w = 2
        rconv5W = tf.Variable(tf.random_normal([k_h,k_w,384,c_o],stddev=0.1))
        rconv5b = tf.Variable(tf.random_normal([c_o],mean= -0.5,stddev= 0.3))
        conv5_in = tf.nn.conv2d(conv4, rconv5W, strides=[1,s_h,s_w,1], padding='VALID')
        conv5_add = tf.nn.bias_add(conv5_in, rconv5b)
        conv5 = tf.nn.relu(conv5_add)


        #fc6
        #fc(4096, name='fc6')
        len_input = int(np.prod(conv5.get_shape()[1:]))
        
        rfc6W = tf.Variable(tf.random_normal([len_input, 4096], stddev=0.1))
        rfc6b = tf.Variable(tf.random_normal([4096], mean= -0.5,stddev= 0.3))
        fc6 = tf.nn.relu_layer(tf.reshape(conv5, [-1, len_input]), rfc6W, rfc6b)
        
        #fc7
        #fc(4096, name='fc7')
        rfc7W = tf.Variable(tf.random_normal([4096, 4096], stddev=0.1))
        rfc7b = tf.Variable(tf.random_normal([4096], mean= -0.5,stddev= 0.3))
        fc7 = tf.nn.relu_layer(fc6, rfc7W, rfc7b)
        
        #fc8
        #fc(1000, relu=False, name='fc8')
        rfc8W = tf.Variable(tf.random_normal([4096, 1000], stddev=0.1))
        rfc8b = tf.Variable(tf.random_normal([1000], mean= -0.5,stddev= 0.3))
        fc8 = tf.nn.xw_plus_b(fc7, rfc8W, rfc8b)     
        prob = tf.nn.softmax(fc8)

        tf.summary.histogram('conv1W', rconv1W)
        tf.summary.histogram('conv1b', rconv1b)
        tf.summary.histogram('conv2W', rconv2W)
        tf.summary.histogram('conv2b', rconv2b)
        tf.summary.histogram('conv3W', rconv3W)
        tf.summary.histogram('conv3b', rconv3b)
        tf.summary.histogram('conv4W', rconv4W)
        tf.summary.histogram('conv4b', rconv4b)
        tf.summary.histogram('conv5W', rconv5W)
        tf.summary.histogram('conv5b', rconv5b)

        tf.summary.histogram('r_fc8w', rfc8W)
        tf.summary.histogram('r_fc8b', rfc8b)
    
prod_loss = tf.subtract(prob, loss_van)
prod_l2n = tf.sqrt(tf.reduce_mean(tf.multiply(prod_loss,prod_loss)))

tf.summary.scalar("prod_l2n",prod_l2n)
var_grad = tf.train.AdamOptimizer(1e-1).compute_gradients(prod_l2n)
train_step = tf.train.AdamOptimizer(1e-1).minimize(prod_l2n)


with tf.Session() as sess:

    filename = "../model/half_2_full/fcann_v1.ckpt"
    logfile = '../log/half_2_full'
    graph_model = '../model/half_2_full/fcann_v1.ckpt-1000.meta'
    checkpoint_dir = '../model/half_2_full'
    
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logfile, sess.graph)  
    
    init = tf.global_variables_initializer()
    sess.run(init)  
    
    
    saver = tf.train.Saver()  
    if continue_training !=0:
        resaver = tf.train.import_meta_graph(graph_model)
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
        for i in range(loop_num,20000000):
            
            print("Training Step: {}".format(i))
            
            image_tensor = sess.run(sample_batch)
            sess.run(train_step, feed_dict = {x:image_tensor})
            #grad = sess.run(var_grad, feed_dict = {x:image_tensor})
            #print("Grad:{}".format(grad))

            if i%500 == 0 and i != 0:
                
                tloss, summary = sess.run([prod_l2n,merged_summary_op], feed_dict = {x:image_tensor}) 
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
                
                print("total loss:{}".format(tloss))
                print("Save Model:{}".format(filename))
                saved_model = saver.save(sess, filename, global_step=i)
           
           
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)
        
    summary_writer.close()
    sess.close()  
   








