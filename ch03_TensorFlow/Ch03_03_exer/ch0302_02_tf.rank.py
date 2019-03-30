import tensorflow as tf # tf-rank.py

b1 = tf.constant(7)
b2 = tf.constant([3,7])
b3 = tf.constant([[3,7],[11,13]])

sess = tf.Session()
print(sess.run(tf.rank(b1)))
print(sess.run(tf.rank(b2)))
print(sess.run(tf.rank(b3)))
# 0
# 1
# 2