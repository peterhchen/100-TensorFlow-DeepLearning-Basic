##########################################
# TensorFlow APIs IN THIS EXAMPLE:
# tf.placeholder(...)
# tf.Variable(...)
# tf.nn.softmax(...)
# tf.nn.sigmoid_cross_entropy_with_logits(...)
# tf.reduce_sum(...)
# tf.reduce_mean(...)
# tf.train.GradientDescentOptimizer(...)
# tf.equal(...)
# tf.argmax(...)
##########################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W_h1 = tf.Variable(tf.random_normal([784, 512]))
b_1 = tf.Variable(tf.random_normal([512]))
h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_1)

W_out = tf.Variable(tf.random_normal([512, 10]))
b_out = tf.Variable(tf.random_normal([10]))
y_ = tf.nn.softmax(tf.matmul(h1, W_out) + b_out)

# COMPARE RESULT WITH mlp-good.py
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y)

loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train the model
with tf.Session() as s:
  s.run(tf.global_variables_initializer())

  for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(100)
    s.run(train_step, feed_dict={x: batch_x, y: batch_y})

    if i % 1000 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch_x, y:batch_y})
      print('step {0} training accuracy: {1}'.format(i,train_accuracy))
