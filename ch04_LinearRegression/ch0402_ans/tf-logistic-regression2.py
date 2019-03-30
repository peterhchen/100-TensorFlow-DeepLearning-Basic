##########################################
# TensorFlow APIs IN THIS EXAMPLE:
# tf.nn.sigmoid(tf.matmul(...))
# tf.Variable(tf.random_normal(...))
# tf.placeholder(...)
# tf.pow(model_y-Y, 2)/(2)
# tf.global_variables_initializer
# tf.initialize_all_variables().run() 
##########################################

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

np.random.seed(0) # generate same random #s

import matplotlib.pyplot as plt
trainsamples = 200
testsamples = 60

#the model, a simple input, a hidden layer (sigmoid activation)
def model(X, hidden_weights1, hidden_bias1, ow):
  hidden_layer = tf.nn.sigmoid(tf.matmul(X, hidden_weights1)+ b)
  return tf.matmul(hidden_layer, ow)

dsX = np.linspace(-1, 1, trainsamples + testsamples).transpose()
#dsY = 2*dsX+np.random.randn(*dsX.shape)*0.5 + 1.0
dsY = 2.0*pow(dsX,2)+2*dsX+np.random.randn(*dsX.shape)*0.22+0.8

#print("dsX:",dsX)
#print("dsY:",dsY)
print("dsX shape:",dsX.shape)
print("dsY shape:",dsY.shape)

X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create first hidden layer
hw1 = tf.Variable(tf.random_normal([1, 10], stddev=0.1))

# Create output connection
ow = tf.Variable(tf.random_normal([10, 1], stddev=0.1))
print("output ow:",ow)

# Create bias
b = tf.Variable(tf.random_normal([10], stddev=0.1))
print("bias b:",b)

model_y = model(X, hw1, b, ow)

# Cost function
cost = tf.pow(model_y-Y, 2)/(2)

# construct an optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# Launch the graph in a session
epochs = 50 

with tf.Session() as sess:
  tf.initialize_all_variables().run() #Initialize all variables
 #tf.global_variables_initializer()   #initialize all variables

  print("-------------------------")
  print("bias b:")
  print(sess.run(b))
  print("output ow:")
  print(sess.run(ow))
  print("-------------------------\n")

  # nested loop for the training data
  for i in range(1,epochs):
    #randomize the samples to implement a better training
    dsX, dsY = shuffle (dsX, dsY) 
    trainX, trainY = dsX[0:trainsamples], dsY[0:trainsamples]

    cost1=0.
    for x1,y1 in zip (trainX, trainY):
      cost1 += sess.run(cost,feed_dict={X:[[x1]], Y:y1}) # /trainsamples
      sess.run(train_op, feed_dict={X: [[x1]], Y: y1})
     #sess.run(train_op, feed_dict={X: x1, Y: y1})
    if (i % 10 == 0):
      print("Average train cost for epoch " + str (i) + ":" + str(cost1))

  # nested loop for the test data
  for i in range(1,epochs):
    testX, testY = dsX[trainsamples:trainsamples + testsamples], dsY[trainsamples:trainsamples+testsamples]

    cost2=0.
    for x1,y1 in zip (testX, testY):
      cost2 += sess.run(cost,feed_dict={X:[[x1]], Y:y1}) # /testsamples
    if (i % 10 == 0):
      print("Average test cost for epoch " + str (i) + ":" + str(cost2))

plt.figure() 
plt.title('Original data')
plt.scatter(dsX,dsY)
plt.show()

