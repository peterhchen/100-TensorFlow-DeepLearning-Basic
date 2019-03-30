import tensorflow as tf # random-normal.py

# initialize a 6x3 array of random numbers:
values = {'weights':tf.Variable(tf.random_normal([6,3]))}

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(values['weights']))