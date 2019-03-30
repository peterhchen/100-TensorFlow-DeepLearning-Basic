import tensorflow as tf

W = tf.Variable([.5], dtype=tf.float32)
b = tf.Variable([-1], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

maxVal = 1000 
with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init) 
  for i in range(maxVal):
    # y = -x + 1 
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    if(i % 50) == 0:
      print(sess.run([W, b]))
  print("W and b:",sess.run([W, b]))

