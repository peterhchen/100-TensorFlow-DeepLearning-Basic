import tensorflow as tf

print("default:", tf.get_default_graph())
print("new one:", tf.Graph())

graph1 = tf.get_default_graph()
graph2 = tf.Graph()

with graph2.as_default():
  print("graph2 is default:")
  print(graph2 is tf.get_default_graph())

