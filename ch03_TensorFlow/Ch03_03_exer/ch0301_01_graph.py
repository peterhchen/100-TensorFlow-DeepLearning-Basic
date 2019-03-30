import tensorflow as tf #basic-buffer.py

g = tf.Graph()
with g.as_default():
   sess = tf.Session()
   a = tf.placeholder('float', name='a')
   b = tf.placeholder('float', name='b')
   c = tf.multiply(a,b, name='c')
   feed_dict = {a:2, b:3}
   print(sess.run(c, feed_dict))
   print (g.as_graph_def())

# => see basic-buffer.out