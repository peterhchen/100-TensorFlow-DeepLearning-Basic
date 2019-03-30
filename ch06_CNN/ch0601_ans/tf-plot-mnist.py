# https://joomik.github.io/MNIST/

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
#%matplotlib inline

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Number of training examples:", mnist.train.num_examples)
print("Number of validation examples:", mnist.validation.num_examples)
print("Number of testing examples:", mnist.test.num_examples)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)

left= 2.5
top = 2.5

fig = plt.figure(figsize=(10,10))

for i in range(6):
  ax = fig.add_subplot(3,2,i+1)
  im = np.reshape(mnist.train.images[i,:], [28,28])

  label = np.argmax(mnist.train.labels[i,:])
  ax.imshow(im, cmap='Greys')
  ax.text(left, top, str(label))

# A placeholder for the data (inputs and outputs)
x = tf.placeholder(tf.float32, [None, 784])

# W: the weights for each pixel for each class
# b: bias of each class
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# The model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# placeholder to input correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# A measure of model precision using cross-entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# minimize cross_entropy with gradient descent with 0.5 as lr
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()

# the execution
sess = tf.Session()
sess.run(init)

# run training step 1000 times
for i in range(1000):
  # get random 100 data samples from the training set
  batch_xs, batch_ys = mnist.train.next_batch(100)
    
  # feed them to the model in place of the placeholders defined above
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#this accuracy returns the mean value of an array of 1s and 0s.
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# retrun the accuracy on the test set.
print("Accuracy: ", sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

correct_vals = sess.run(correct_pred, 
                        feed_dict={x: mnist.train.images, y_: mnist.train.labels})
pred_vals = sess.run(y, feed_dict={x: mnist.train.images} )

sess.close()

cntFalse = 0
for cv in correct_vals:
    if cv==False:
        cntFalse+=1
print(cntFalse, "incorrect labels out of",  len(correct_vals))

fig = plt.figure(figsize=(10,10))

cntFalse = 0
for i, cv in enumerate(correct_vals):
  if cv==False:
    cntFalse +=1

    ax = fig.add_subplot(3,2,cntFalse)
    im = np.reshape(mnist.train.images[i,:], [28,28])

    label = np.argmax(mnist.train.labels[i,:])
    pred_label = np.argmax(pred_vals[i,:])
    
    ax.imshow(im, cmap='Greys')
    ax.text(left, top, 'true=' + str(label) + ', pred=' + str(pred_label))
    
  if cntFalse==6:
    break

plt.show()
