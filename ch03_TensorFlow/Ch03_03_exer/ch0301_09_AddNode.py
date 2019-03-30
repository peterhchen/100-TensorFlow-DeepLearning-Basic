import tensorflow as tf #addnodes.py

a1 = tf.placeholder(tf.float32)
a2 = tf.placeholder(tf.float32)
a3 = a1 + a2

sess = tf.Session()
print(sess.run(a3, {a1:7, a2:13}))
print(sess.run(a3, {a1:[7,10], a2:[13,20]}))

# 20.0
# [20. 30.]