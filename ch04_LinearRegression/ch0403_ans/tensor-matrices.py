import tensorflow as tf

#error:
#a1 = tf.constant([1,2],[3,4])

a1 = tf.constant([[1,2],[3,4]])
print("a1 shape:",a1.get_shape())

a2 = tf.constant([[10],[20]])
print("a2 shape:",a2.get_shape())

a3 = tf.Variable(initial_value=a2)
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  print("a3:",sess.run(a3))

