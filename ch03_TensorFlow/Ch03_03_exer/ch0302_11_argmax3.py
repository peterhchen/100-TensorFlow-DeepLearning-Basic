import tensorflow as tf # tf-argmax2.py
import numpy as np
x = np.array([[31, 23,  4, 54],
              [18,  3, 25,  0],
              [28, 14, 33, 22],
              [17, 12,  5, 81]])
y = np.array([[31, 23,  4, 24],
              [18,  3, 25,  0],
              [28, 14, 33, 22],
              [17, 12,  5, 11]])
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print('xmax:', sess.run(tf.argmax(x,1)))
  print('ymax:', sess.run(tf.argmax(y,1)))
  print('equal:',sess.run(tf.equal(x,y)))