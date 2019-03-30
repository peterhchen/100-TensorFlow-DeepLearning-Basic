import tensorflow as tf # tf-reshape-eval.py

x=tf.constant([[2,5,3,-5],[0,3,-2,5],[4,3,5,3]])

sess = tf.Session()
print(sess.run(tf.shape(x)))
print('1:',sess.run(tf.reshape(x, [6,2])))

with sess.as_default():
  print('2:',tf.reshape(x, [3,4]).eval())
