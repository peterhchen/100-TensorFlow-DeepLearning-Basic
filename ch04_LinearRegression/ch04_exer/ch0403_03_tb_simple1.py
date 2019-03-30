import tensorflow as tf

x = tf.constant(35, name='x')
print(x)
y = tf.Variable(x + 5, name='y')

with tf.Session() as session:
  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter("simple", session.graph)
  init =  tf.global_variables_initializer()
  session.run(init)
  print(session.run(y))