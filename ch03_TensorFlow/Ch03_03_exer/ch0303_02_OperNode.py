import tensorflow as tf # linreg1.py

W = tf.Variable([0.5, 0.3],tf.float32)
b = tf.Variable([3.6, 6.0],tf.float32)
x = tf.placeholder(tf.float32)
lm = W * x + b

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(lm, {x:[8,10]}))
# => [7.6 9. ]