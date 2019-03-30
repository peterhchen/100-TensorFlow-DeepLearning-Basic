import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n_features = 10
n_neurons  = 3

# z = x*W + b:  
# [None,n_features] x [n_features,n_neurons] + 1 x n_neurons 

x = tf.placeholder(tf.float32, (None,n_features))
W = tf.Variable(tf.random_normal([n_features,n_neurons])) 
b = tf.Variable(tf.ones([n_neurons]))
z = tf.add(tf.matmul(x,W), b)

################################
#sigmoid, tanh, relu, relu6, etc 
#a = tf.sigmoid(z)
#a = tf.nn.relu(z)
#a = tf.tanh(z)
################################

# initialize m and b with random values 
arr1 = np.random.rand(2) # arr1 = (m,b)
m = tf.Variable(arr1[0])
b = tf.Variable(arr1[1])

# initialize some data 
x_data  = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

# set up error function 
error = 0
for x,y in zip(x_data, y_label):
  y_hat = m*x + b # predicted value
  error += (y-y_hat)**2

# set up optimizer
lr = 0.001 
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op  = optimizer.minimize(error)

training_steps = 500
with tf.Session() as sess:
  tf.global_variables_initializer().run()

  for i in range(training_steps):
    sess.run(train_op)
 
  final_m, final_b = sess.run([m,b])

print("m: ",arr1[0],"b: ",arr1[1])
print("fm:",final_m,"fb:",final_b)

x_test = np.linspace(-1,11,10)
y_pred_plot = final_m*x_test + final_b

# plot best-fitting line:
plt.plot(x_test, y_pred_plot, 'r')

# initial random points: 
plt.plot(x_data, y_label, '*')
plt.show()

