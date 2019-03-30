import tensorflow as tf # tf-rotate-image1.py
import matplotlib.image as img
import matplotlib.pyplot as plot

myfile = "./dandelion.png"
myimage = img.imread(myfile)
image = tf.Variable(myimage,name='image')
vars = tf.global_variables_initializer()

sess = tf.Session()
flipped = tf.transpose(image, perm=[1,0,2])
sess.run(vars)
result=sess.run(flipped)
plot.imshow(result)
plot.show()