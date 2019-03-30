import tensorflow as tf
import numpy as np

n_features = 10
n_neurons  = 3

# z = x*W + b:  
# [None,n_features] x [n_features,n_neurons] + 1 x n_neurons 

x = tf.placeholder(tf.float32, (None,n_features))
W = tf.Variable(tf.random_normal([n_features,n_neurons])) 
b = tf.Variable(tf.ones([n_neurons]))
z = tf.add(tf.matmul(x,W), b)

################################
#sigmoid, tanh, relu, relu6, etc 
a = tf.sigmoid(z)
#a = tf.nn.relu(z)
#a = tf.tanh(z)
################################

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  out = sess.run(a, feed_dict={x:np.random.random([1,n_features])})
  print("values:",out) 
