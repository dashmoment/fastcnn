import tensorflow as tf
import numpy as np


rconv1_w = tf.Variable(tf.random_normal([11,11,3,96]), name="cw1")

new_graph = tf.Graph()

train_x = np.zeros((1,227,227,3)).astype(np.float32)
train_y = np.zeros((1,1000))
x_dim = train_x.shape[1:]
y_dim = train_y.shape[1:]

x = tf.placeholder(tf.float32,(None,) + x_dim)


#Load trained model
net_data = np.load("bvlc_alexnet.npy", encoding='latin1').item()

#conv1

k_size = 11
stride = 4
out_size = 96



with tf.Session(graph=new_graph) as sess:
    conv1_w = tf.Variable(net_data["conv1"][0])
    conv1_b = tf.Variable(net_data["conv1"][1])
    new_saver = tf.train.import_meta_graph('./model/fcann_v1.ckpt-0.meta')
    var2 = new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    tw1 = sess.run(tf.get_collection("w1")[0])
    tb1 = sess.run(tf.get_collection("b1")[0])
    ow1 = sess.run(conv1_w)
    ob1 = sess.run(conv1_b)
    
    
    