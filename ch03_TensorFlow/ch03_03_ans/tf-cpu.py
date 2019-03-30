import tensorflow as tf # tf-cpu.py

with tf.Session() as sess:
  m1 = tf.constant([[3., 3.]])
  m2 = tf.constant([[2.],[2.]])
  p1 = tf.matmul(m1, m2)
  print('m1:',sess.run(m1))
  print('m2:',sess.run(m2))
  print('p1:',sess.run(p1))

#('m1:', array([[3., 3.]],  dtype=float32))
#('m2:', array([[2.], [2.]],dtype=float32))
#('p1:', array([[12.]],     dtype=float32))
