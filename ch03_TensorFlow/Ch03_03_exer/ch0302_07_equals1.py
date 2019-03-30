import tensorflow as tf # tf-equal1.py

x1 = tf.constant([0.9, 2.5, 2.3, -4.5])
x2 = tf.constant([1.0, 2.0, 2.0, -4.0])
eq = tf.equal(x1,x2)

with tf.Session() as sess:
  print('x1:',sess.run(x1))
  print('x2:',sess.run(x2))
  print('eq:',sess.run(eq))