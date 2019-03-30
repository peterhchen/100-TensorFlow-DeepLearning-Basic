import tensorflow as tf # tf-range1.py

a1 = tf.range(3, 18, 3)
a2 = tf.range(0, 8, 2)
a3 = tf.range(-6, 6, 3)
a4 = tf.range(-10, 10, 4)

with tf.Session() as sess:
  print('a1:',sess.run(a1))
  print('a2:',sess.run(a2))
  print('a3:',sess.run(a3))
  print('a4:',sess.run(a4))