##########################################
# TensorFlow APIs IN THIS EXAMPLE:
# tf.nn.sigmoid(tf.matmul(...))
##########################################

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

np.random.seed(0) # generate same random #s

import matplotlib.pyplot as plt
trainsamples = 200
testsamples = 60

#the model, a simple input, a hidden layer (sigmoid activation)
#NOTE: this model is not used in this code sample
def model(X, hidden_weights1, hidden_bias1, ow):
  hidden_layer =  tf.nn.sigmoid(tf.matmul(X, hidden_weights1)+ b)
  return tf.matmul(hidden_layer, ow)

dsX = np.linspace(-1, 1, trainsamples + testsamples).transpose()
dsY = 2*dsX+np.random.randn(*dsX.shape)*0.2
#dsY = 0.4*pow(dsX,2)+2*dsX+np.random.randn(*dsX.shape)*0.22+0.8

plt.figure() 
plt.title('Original data')
plt.scatter(dsX,dsY)
plt.show()