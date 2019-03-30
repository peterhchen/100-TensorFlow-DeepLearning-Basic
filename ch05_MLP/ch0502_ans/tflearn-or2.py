# https://towardsdatascience.com/tflearn-soving-xor-with-a-2x2x1-feed-forward-neural-network-6c07d88689ed
# XOR gate OK requires:
# 1. Two hinnde layers
# 2. Actionvation function by tanh
# 3. loss funciotn by binary_crossentropy
from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#Training examples
X = [[0,0], [0,1], [1,0], [1,1]]
#AND: OK
#Y = [[0], [0], [0], [1]]
#XAND/XNOR: OK
#Y = [[1], [0], [0], [1]]
#NAND: OK
#Y = [[1], [1], [1], [0]]
#OR: OK
Y = [[0], [1], [1], [1]]
#NOR: OK
#Y = [[1], [0], [0], [0]]
# XOR/XNAND:OK
#Y = [[0], [1], [1], [0]]

input_layer = input_data(shape=[None, 2]) #input layer of size 2
hidden_layer = fully_connected(input_layer , 2, activation='tanh') #hidden layer of size 2
output_layer = fully_connected(hidden_layer, 1, activation='tanh') #output layer of size 1

#use Stohastic Gradient Descent and Binary Crossentropy as loss function
regression = regression(output_layer , optimizer='sgd', loss='binary_crossentropy', learning_rate=5)
model = DNN(regression)

#fit the model
model.fit(X, Y, n_epoch=2000, show_metric=True);

#predict all examples
print ('Input Logic:  ', [i[0] > 0 for i in X])
print ('Expected Outpout Data:  ', Y)
print ('Predicted Output Data: ', model.predict(X))
print ('Expected Outpout Logic:  ', [i[0] > 0 for i in Y])
print ('Predicted Output Logic: ', [i[0] > 0 for i in model.predict(X)])