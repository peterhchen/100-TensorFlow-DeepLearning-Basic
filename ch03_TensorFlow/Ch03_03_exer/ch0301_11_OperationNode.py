import tensorflow as tf #addnodes3.py
a1 = tf.placeholder(tf.float32)
a2 = tf.placeholder(tf.float32)
a3 = a1 + a2
a4 = a3*6
a5 = a4/2

with tf.Session() as sess: #addnodes3.py
    print(sess.run(a4, {a1:7, a2:13}))
    print(sess.run(a5, {a1:[7,10], a2:[13,20]}))
#  print('a4: ',sess.run(a4))
#   print('a5: ',sess.run(a5))

#   b1 = tf.add_n([a4, a5], name="b1")
#   print('b1: ',sess.run(b1))

#   b2 = tf.multiply(a4, a5, name="b2")
#   print('b2: ',sess.run(b2))

#   b3 = tf.multiply(tf.pow(a4,2),tf.pow(a5,2), name="b3")
#   print('b3: ',sess.run(b3))