#https://itnext.io/how-to-use-tensorboard-5d82f8654496
import tensorflow as tf

# clear the defined variables and operations of previous cell
tf.reset_default_graph()   

# create the scalar variable
x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

# step 1: create the scalar summary
first_summary = tf.summary.scalar(name='My_first_scalar_summary', tensor=x_scalar)
init = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
  # step 2: creating the writer inside the session
  writer = tf.summary.FileWriter('./summary1', sess.graph)

  for step in range(100):
    # loop over several initializations of the variable
    sess.run(init)

    # step 3: evaluate the scalar summary
    summary = sess.run(first_summary)

    # step 4: add the summary to the writer (i.e. to the event file)
    writer.add_summary(summary, step)

  print('Done with writing the scalar summary')

