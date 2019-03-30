import tensorflow as tf #addnodes2.py

a1 = tf.placeholder(tf.float32)
a2 = tf.placeholder(tf.float32)
a3 = a1 + a2
a4 = a3*6
a5 = a4/2

sess = tf.Session()
print(sess.run(a4, {a1:7, a2:13}))
print(sess.run(a5, {a1:[7,10], a2:[13,20]}))

# 120.0
# [60. 90.]