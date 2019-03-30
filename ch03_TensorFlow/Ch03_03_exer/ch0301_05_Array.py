import os # array1.py
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

# create a Python array:
array_1d = np.array([1.3, 1, 4.0, 23.5])

tf_tensor = \
    tf.convert_to_tensor(array_1d, dtype=tf.float64)
with tf.Session() as sess:
  print (sess.run(tf_tensor))
  print (sess.run(tf_tensor[0]))
  print (sess.run(tf_tensor[2]))