import tensorflow as tf; import numpy as np; import matplotlib.pyplot as plt
tf.set_random_seed(0)
np.random.seed(42)
# for sigmoid activation 
m1 = tf.Variable(tf.random_normal(shape=[1,1]))
b1 = tf.Variable(tf.random_uniform(shape=[1,1]))
step = 50 
count = 500
batch_size = int(count/4)
# normal distribution (mean/stddev/# of points) 
x = np.random.normal(2, 0.1, count)
#print("x:",x)
# NOTE: x_data is a COLUMN vector:
x_data = tf.placeholder(shape=[None,1], dtype=tf.float32,name='x_data')
linear1  = tf.add(tf.matmul(x_data,m1), b1)
sigmoid1 = tf.sigmoid(tf.add(tf.matmul(x_data,m1), b1))
loss1    = tf.reduce_mean(tf.square(tf.subtract(sigmoid1,0.75)))
lrate    = 0.01
optimzr  = tf.train.GradientDescentOptimizer(lrate)
init     = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  train_sigmoid1 = optimzr.minimize(loss1)
  loss_sigmoid1  = []
  for i in range(count):
    rand_indices = np.random.choice(len(x), size=batch_size)\
    # convert from column data to row data:
    x_vals = np.transpose([x[rand_indices]])
    # => if you remove 'transpose' above you will get an error:
    # Cannot feed shape (125,)... to Tensor with shape '(?, 1)'
    # x_vals = x[rand_indices]
    # train the model:
    sess.run(train_sigmoid1, feed_dict={x_data : x_vals}) 
    out_sig1 = np.mean(sess.run(sigmoid1,feed_dict={x_data:x_vals}))
    # store loss information to display in a plot:
    loss_sigmoid1.append(sess.run(loss1,feed_dict={x_data:x_vals}))
    if( i == 0):
       print("sigmoid1 = x_data*m1 + b1")
       print("x_data shape:",x_data)
       print("m1     shape:",m1.shape)
       print("b1     shape:",b1.shape)
       print("x_vals shape:",x_vals.shape)
    if( (i % step) == 0):
       print("sigmoid1:",out_sig1)
  plt.plot(loss_sigmoid1, 'k-',  label='Sigmoid'); plt.ylim([0, 1.0])
  plt.title("Loss per Iteration"); plt.xlabel("Iteration"); plt.ylabel("Loss") 
  plt.legend(loc='upper right'); plt.show()