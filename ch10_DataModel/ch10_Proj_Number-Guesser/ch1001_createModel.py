import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

for train in range(len(x_train)):
#for train in range(2):
    # print ('train: ', train)
    for row in range(28):
        #print ('row: ', row)
        for x in range(28):
            # print ('x: ', x)
            if x_train[train][row][x] != 0:
                x_train[train][row][x] = 1
            #print ('x_train [', train, '][', row, '][', x, '] = ', x_train[train][row][x])

#print ('shape: ', tf.shape (x_train))
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
model.save('epic_num_reader.model')


# FOR TESTING
'''
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)

for test in range(len(x_test)):
    for row in range(28):
        for x in range(28):
            if x_test[test][row][x] != 0:
                x_test[test][row][x] = 1


model = tf.keras.models.load_model('epic_num_reader.model')

predictions = model.predict(x_test)

count = 0
for x in range(len(predictions)):
    guess = (np.argmax(predictions[x]))
    actual = y_test[x]
    #print("I predict this number is a:", guess)
    #print("Number Actually Is a:", actual)
    if guess != actual:
        #print("--------------")
        #print('WRONG')
        #print('---------------')
        count+=1
    #plt.imshow(x_test[x], cmap=plt.cm.binary)
    #plt.show()
    #input("Press enter to show next number")

print("The program got", count, 'wrong, out of', len(x_test))
print(str(100 - ((count/len(x_test))*100)) + '% correct')

'''