import tensorflow as tf # tb-pholders2.py

vect = [1,2,3,4]
a = tf.placeholder(tf.float32, shape=[])
b = tf.constant([3], tf.float32)
c = a + b 

with tf.Session() as sess:
  for a_val in vect:
    print(sess.run(c, {a: a_val}))

  writer = tf.summary.FileWriter("pholders2", sess.graph)
  writer.add_graph(sess.graph) 
  writer.close()

# tensorboard --logdir=pholders2