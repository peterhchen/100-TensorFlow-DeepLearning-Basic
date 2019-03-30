import tensorflow as tf

lr   = tf.Variable(0.01, name='lr')
loss = tf.Variable(0.55, name='loss')

with tf.Session() as sess:
  writer = tf.summary.FileWriter("minimal2", sess.graph)
  writer.add_graph(sess.graph) 
  tf.summary.scalar("lr", lr)
  tf.summary.scalar("loss", loss)

# tensorboard --logdir=minimal2

