import tensorflow as tf # tf-linspace1.py
import numpy as np

np.random.seed(0) # generate same random #s

trainX = np.linspace(-1, 1, 6)
trainY = 3*trainX+ np.random.randn(*trainX.shape)*0.5

print("trainX: ", trainX)
print("trainY: ", trainY)

with tf.Session() as sess:
 print("trainX: ", trainX)
 print("trainY: ", trainY)