##########################################
# TensorFlow APIs IN THIS EXAMPLE:
# tf.placeholder(...)
# tf.Variable(tf.zeros(...))
# tf.multiply(...)
# tf.square(...)
# tf.train.GradientDescentOptimizer(...)
##########################################

import tensorflow as tf
import numpy as np

np.random.seed(0) # generate same random #s

# x and y are placeholders for training data
x = tf.placeholder("float")
y = tf.placeholder("float")

# w stores our values: initialised with starting "guesses"
# w[0] and w[1] are "a" and "b"
w = tf.Variable([1.0, 2.0], name="w")

# model: y = a*x + b
y_model = tf.multiply(x, w[0]) + w[1]

# error = the square of the differences
error = tf.square(y - y_model)

# Gradient Descent Optimizer (for the heavy lifting)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

model = tf.global_variables_initializer()

with tf.Session() as session:
  session.run(model)
  for i in range(1000):
    x_value = np.random.rand()
    y_value = x_value * 2 + 6
    session.run(train_op, feed_dict={x: x_value, y: y_value})

  w_value = session.run(w)
  print("Predicted: {a:.3f}x + {b:.3f}".format(a=w_value[0],b=w_value[1]))

