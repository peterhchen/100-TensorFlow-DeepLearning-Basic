import tensorflow as tf

with tf.Session() as sess:
  writer = tf.summary.FileWriter("minimal", sess.graph)
  writer.add_graph(sess.graph) 

# tensorboard --logdir=minimal
# NB: nothing is generated y