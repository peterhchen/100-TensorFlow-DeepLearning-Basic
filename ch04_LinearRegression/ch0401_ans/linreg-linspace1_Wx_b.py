import tensorflow as tf
import numpy as np 

np.random.seed(0) # generate same random #s

W = tf.Variable([0.5], dtype=tf.float32)
b = tf.Variable([3.6], dtype=tf.float32)
x = tf.placeholder(tf.float32)

# the linear model:
lm = W * x + b

y = tf.placeholder(tf.float32)
sq_deltas = tf.square(lm - y)
loss = tf.reduce_sum(sq_deltas)
lr = 0.001

# specify the following:
# 1) loss function (optimizer)
# 2) learning rate
# 3) minimize loss function
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)

# experiment with these values:
factor = 0.25 
x_train = np.linspace(-1, 1, 6) # 6 spaced numbers
y_train = 3*x_train+ np.random.randn(*x_train.shape)*factor

init = tf.global_variables_initializer()

# experiment with these values:
iterations = 1000 # 100, 500, 1000  
threshold  = 100  # 10,  50,  100 

with tf.Session() as sess:
  sess.run(init) 

  for i in range(iterations):
    if (i+1) % threshold == 0:
      print(sess.run([W, b]))
    sess.run(train, {x:x_train, y:y_train})

  print('W:',sess.run([W]))
  print('b:',sess.run([b]))

