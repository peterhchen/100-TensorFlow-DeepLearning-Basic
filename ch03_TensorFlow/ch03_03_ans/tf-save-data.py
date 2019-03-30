import tensorflow as tf # tf-save-data.py
x = tf.constant(5,name="x")
y = tf.constant(8,name="y")
z = tf.Variable(2*x+3*y, name="z")
model = tf.global_variables_initializer()

with tf.Session() as session:
  writer = tf.summary.FileWriter("./tf_logs",session.graph)
  session.run(model)
  print('z = ',session.run(z)) # =>  z = 34

# launch tensorboard: tensorboard hyphenhyphen logdir=./tf_logs

