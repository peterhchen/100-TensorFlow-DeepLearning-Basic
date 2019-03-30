import tensorflow as tf # linear1.py

# W and x are one-d arrays
W = tf.constant([20,30],     name='W')
x = tf.placeholder(tf.int32, name='x')
b = tf.placeholder(tf.int32, name='b')

Wx = tf.multiply(W, x, name='Wx')
y  = tf.add(Wx, b, name='y')

with tf.Session() as sess:
  print("Result 1: Wx = ", 
   sess.run(Wx, feed_dict={x:[5,10]}))
  print("Result 2: y  = ", 
   sess.run(y, feed_dict={x:[5,10],b:[15,25]}))

# Result 1: Wx = [100 300]
# Result 2: y  = [115 325]

