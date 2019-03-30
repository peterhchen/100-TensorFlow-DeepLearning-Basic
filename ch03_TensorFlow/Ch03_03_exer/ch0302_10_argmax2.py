import tensorflow as tf # tf-row-max.py

# initialize array of arrays:
a = [[1,2,3], [30,20,10], [40,60,50]]
b = tf.Variable(a, name='b')

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print("index of max values in b: ",
          sess.run(tf.argmax(b,1)))
#('index of max values in b: ',array([2, 0, 1]))