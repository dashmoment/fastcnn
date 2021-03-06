import tensorflow as tf
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy import misc
import purgeinvalid_img as pi
import net_factory



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



pi.purgeinvalidandRGB_img("../../dataset/ilsvrc_train") #Cleanup Invalid file in directory

train_x = np.zeros((1,227,227,3)).astype(np.float32)
train_y = np.zeros((1,1000))
x_dim = train_x.shape[1:]
y_dim = train_y.shape[1:]

x = tf.placeholder(tf.float32,(None,) + x_dim)


#Load trained model
net_data = np.load("../model/bvlc_alexnet.npy", encoding='latin1').item()

#conv1

k_size = 11
stride = 4
continue_training = 1
loop_num = 13045


with tf.name_scope("conv1"):

    conv1_w = tf.Variable(net_data["conv1"][0], trainable = False)
    conv1_b = tf.Variable(net_data["conv1"][1], trainable = False)
    conv1_in = tf.nn.conv2d(x,conv1_w, strides=[1,stride,stride,1], padding='SAME')
    conv_in = tf.nn.bias_add(conv1_in, conv1_b)
    conv1 = tf.nn.relu(conv_in)
    
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    tf.summary.histogram('conv', conv1_w)
    tf.summary.histogram('bias', conv1_b)
    


with tf.name_scope("New_conv1"):

     
    rconv1_w = tf.Variable(tf.random_normal([11,11,3,96],stddev=0.1))
    rconv1_b = tf.Variable(tf.random_normal([96],mean= -0.5,stddev= 0.3))

    rconv1_in = tf.nn.conv2d(x, rconv1_w, strides=[1,8,8,1], padding='VALID')
    rconv_in = tf.nn.bias_add(rconv1_in, rconv1_b)
    rconv1_relu = tf.nn.relu(rconv_in)
    rlrn1 = tf.nn.local_response_normalization(rconv1_relu,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    tf.summary.histogram('r_conv', rconv1_w)
    tf.summary.histogram('r_bias', rconv1_b)
    
    
    
def commonlayer(input_l):
    
    with tf.name_scope("Common"):
        
        k_h = 5; k_w = 5; s_h = 1; s_w = 1;c_o = 256;group = 2
        conv2W = tf.Variable(net_data["conv2"][0], trainable = False)
        conv2b = tf.Variable(net_data["conv2"][1], trainable = False)
        
        conv2_in = net_factory.conv(input_l, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        #conv2_in = tf.nn.conv2d(input_l,conv2W, strides=[1,stride,stride,1], padding='SAME')
       
        conv2_add = tf.nn.bias_add(conv2_in, conv2b)
        conv2 = tf.nn.relu(conv2_add)
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
        k_h = 3; k_w = 3; s_h = 2; s_w = 2
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
        
        
    return maxpool2

    


raw = tf.placeholder(tf.float32,(None,) + x_dim)
image_submean = tf.subtract(raw, tf.reduce_mean(raw))
       
pool_loss = tf.subtract(maxpool1, rlrn1)
pool_res = tf.reduce_sum(tf.multiply(pool_loss,pool_loss))

conv_res = pool_res

maxpool2 = commonlayer(maxpool1)
rmaxpool2 = commonlayer(rlrn1)
pool2_loss = tf.subtract(maxpool2, rmaxpool2)
pool2_res = tf.reduce_mean(tf.multiply(pool2_loss,pool2_loss))

total_res = conv_res + pool_res
total_res = pool_res
tf.summary.scalar("pool_loss",pool_res)
tf.summary.scalar("conv_loss",conv_res)
tf.summary.scalar("total_res",total_res)
tf.summary.scalar("pool2_loss",pool2_res)

sample_batch = randombatch()

train_loss = pool_res + pool2_res
train_step = tf.train.AdamOptimizer(1e-4).minimize(pool2_res)


with tf.Session() as sess:

    filename = "../model/half_2_1e-4_dstride_p2/fcann_v1.ckpt"
    logfile = '../log/half_2_1e-4_dstride_p2'
    graph_model = '../model/half_2_1e-4_dstride_p2/fcann_v1.ckpt-16000.meta'
    checkpoint_dir = '../model/half_2_1e-4_dstride_p2'
    
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logfile, sess.graph)  
    
    init = tf.global_variables_initializer()
    sess.run(init)  
    
    
    saver = tf.train.Saver()  
    if continue_training !=0:
        resaver = tf.train.import_meta_graph(graph_model)
        #resaver.restore(sess, '../model/fcann_v1.ckpt-30')   
        resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        sess.run(rconv1_w.assign(tf.get_collection('w1')[0]))
        sess.run(rconv1_b.assign(tf.get_collection('b1')[0]))
        continue_training = 0
    
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord) 
    
    try:
        #while not coord.should_stop():
        for i in range(loop_num,20000000):
            
            
            image_tensor = sess.run(sample_batch)
            
            print("Training Step: {}".format(i))
            print(image_tensor[0][0][0][0])
            
            resimage = sess.run(image_submean, feed_dict={raw:image_tensor})   
    
            sess.run(train_step, feed_dict = {x:resimage})
            
            
            if i%500 == 0:
                
                pres,cres, summary = sess.run([pool_res, conv_res, merged_summary_op], feed_dict = {x:resimage}) 
                isummary = sess.run(tf.summary.image("batch{}".format(i), sample_batch, max_outputs=3))
                
                summary_writer.add_summary(summary, i)
                summary_writer.add_summary(isummary, i)
                
                print("Pool loss:{} , Conv loss:{}".format(pres, cres))
                
                tf.add_to_collection("w1", rconv1_w)
                tf.add_to_collection("b1", rconv1_b)
                tf.add_to_collection("ow1", conv1_w)
                tf.add_to_collection("ob1", conv1_b)
                
                
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
   








