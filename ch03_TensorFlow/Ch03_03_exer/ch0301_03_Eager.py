import tensorflow as tf # tf-func-eager3.py
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

def square(x):
  return tf.multiply(x, x)

grad = tfe.gradients_function(square)

print(square(3.)) # [9.]
print(grad(3.))   # [6.]