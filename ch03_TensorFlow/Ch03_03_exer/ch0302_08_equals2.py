import tensorflow as tf # tf-equal2.py 
import numpy as np

x1 = tf.constant([0.9, 2.5, 2.3, -4.5])
x2 = tf.constant([1.0, 2.0, 2.0, -4.0])
x3 = tf.Variable(x1)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print('x1:',sess.run(x1))
  print('x2:',sess.run(x2))
  print('x3 = round :',sess.run(tf.round(x3)))
  print('eq:',sess.run(tf.equal(x1,x3)))