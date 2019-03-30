import tensorflow as tf # tf-eval.py

a = tf.constant([8], tf.int32, name="a")
x = tf.placeholder(tf.int32, name="x")

y = a * x
with tf.Session() as sess:
  print('y:',y.eval(feed_dict={x:[3]}))

#('y:', array([24], dtype=int32))