import tensorflow as tf # restore-graph.py
x = tf.constant(5,name="x")
y = tf.constant(8,name="y")
z = tf.Variable(2*x+3*y, name="z")
init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(init)
  saver.restore(sess, "model.ckpt")

  # Check values of variables:
  print("z : %s" % z.eval())