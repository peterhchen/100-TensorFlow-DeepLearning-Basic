import tensorflow as tf # tf-show-mnist-images.py
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#1) color version of image #1:
#plt.imshow(mnist.train.images[0].reshape((28, 28), order='C'), interpolation='nearest')

#2) b&w version of image #1:
plt.imshow(mnist.train.images[0].reshape((28, 28), order='C'), cmap='Greys', interpolation='nearest')
plt.show()