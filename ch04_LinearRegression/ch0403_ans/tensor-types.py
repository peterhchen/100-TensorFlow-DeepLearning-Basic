import tensorflow as tf

a1 = tf.constant(5)
a2 = tf.constant([5])
a3 = tf.constant([5,])
b1 = tf.constant([5,9,])
c1 = tf.constant([[5,9],[12, 25,],])
s1 = tf.constant("hello")

print("a1:",a1)
print("a2:",a2)
print("a3:",a3)
print("b1:",b1)
print("c1:",c1)
print("s1:",s1)
print("")

print("a1+a1:",a1+a1)
print("a1+a2:",a1+a2)
print("a1+a3:",a1+a3)
print("a1+b1:",a1+b1)
print("a1+c1:",a1+c1)
print("")

#error:
#c2 = tf.constant([[5,9,7],[12, 25,],])

sess = tf.Session() 
print("a1:",sess.run(a1))
print("a2:",sess.run(a2))
print("a3:",sess.run(a3))
print("b1:",sess.run(b1))
print("c1:",sess.run(c1))
print("s1:",sess.run(s1))
print("")
