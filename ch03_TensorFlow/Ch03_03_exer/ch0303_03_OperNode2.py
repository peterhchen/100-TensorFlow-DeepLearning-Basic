import tensorflow as tf #linreg2.py

W = tf.Variable([0.5, 0.3],tf.float32)
b = tf.Variable([3.6, 6.0],tf.float32)
x = tf.placeholder(tf.float32)
lm = W * x + b

y = tf.placeholder(tf.float32)
sq_deltas = tf.square(lm - y)
loss = tf.reduce_sum(sq_deltas)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print('sess.run: ', sess.run(lm, {x:[1,2], y:[-0.5, 2]}))
print('sq_deltas: ', sess.run (sq_deltas, {x:[1,2], y:[-0.5, 2]}))
print('loss: ', sess.run (loss, {x:[1,2], y:[-0.5, 2]}))