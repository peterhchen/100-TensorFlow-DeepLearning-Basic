# https://stackoverflow.com/questions/41783136/tensorflow-batch-size-in-input-placholder
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  y = tf.nn.softmax(tf.matmul(x,W) + b)
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  
  for i in range(1000):
    batch = mnist.train.next_batch(50) # SET THE BATCH SIZE 
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    if(i % 50 == 0):
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      print(accuracy.eval(feed_dict={x: mnist.test.images,y_: mnist.test.labels}))

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  print(accuracy.eval(feed_dict={x: mnist.test.images,y_: mnist.test.labels}))

