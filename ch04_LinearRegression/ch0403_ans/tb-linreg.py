import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(0) # generate same random #s

# input data:
x_input = np.linspace(0,10,100)
y_input = 5*x_input+2.5

# model parameters and bias:
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

with tf.name_scope('input'):
 X = tf.placeholder(tf.float32, name='InputX')
 Y = tf.placeholder(tf.float32, name='InputY')

# model
with tf.name_scope('model'):
 Y_pred = tf.add(tf.multiply(X,W),b)

# loss
with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.square(Y_pred -Y ))

# training algorithm
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# initializing the variables 
init=tf.global_variables_initializer()

epoch = 50 # 2000
with tf.Session() as sess:
  sess.run(init)
  cost = tf.summary.scalar("loss", loss)

  merged_summary_op = tf.summary.merge_all()
  summ_writer = tf.summary.FileWriter('linreg',graph=tf.get_default_graph())

  for step in range(epoch):
    _, c, summary = sess.run([train, loss, merged_summary_op], \
                             feed_dict={X: x_input, Y: y_input})

    summ_writer.add_summary(summary,step)
    if step % 50 == 0:
      print("cost:", c)

  print("Calculated Model Parameters:") 
  print("Weight: %f" %sess.run(W))
  print("Bias:   %f" %sess.run(b))

