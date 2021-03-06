#https://itnext.io/how-to-use-tensorboard-5d82f8654496
import tensorflow as tf

# clear the defined variables and operations of previous cell
tf.reset_default_graph()   

# create graph
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
c = tf.add(a, b, name="addition")

# creating the writer out of the session
# writer = tf.summary.FileWriter('./tboard2', tf.get_default_graph())

# launch the graph in a session
with tf.Session() as sess:
  # you can create the writer inside the session
  writer = tf.summary.FileWriter('./graphs', sess.graph)
  print(sess.run(c))

