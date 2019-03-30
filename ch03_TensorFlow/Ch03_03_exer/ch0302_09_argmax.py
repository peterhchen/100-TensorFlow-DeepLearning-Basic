import tensorflow as tf # tf-argmax1.py
import numpy as np

x1 = tf.constant([3.9, 2.1, 2.3, -4.0])
x2 = tf.constant([1.0, 2.0, 5.0, -4.2])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print('x1:',sess.run(x1))
  print('x2:',sess.run(x2))

  print('a1:',sess.run(tf.argmax(x1, 0)))
  print('a2:',sess.run(tf.argmax(x2, 0)))