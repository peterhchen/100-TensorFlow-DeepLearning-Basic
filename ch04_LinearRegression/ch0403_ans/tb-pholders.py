import tensorflow as tf # tb-pholders.py

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b 

with tf.Session() as sess:
  print(sess.run(c, {a: [1, 2, 3]}))
  writer = tf.summary.FileWriter("pholders", sess.graph)
  writer.add_graph(sess.graph) 
  writer.close()

# tensorboard --logdir=pholders

