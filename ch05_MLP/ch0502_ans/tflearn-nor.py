import tflearn #tflearn-nor.py 

# Logical NOR operator and "truth" values:
NOR = [[0., 0.],[0., 1.],[1., 0.],[1., 1.]]
Y_truth = [[1.], [0.], [0.], [1.]]

neural_net = tflearn.input_data(shape=[None, 2])
neural_net = tflearn.fully_connected(neural_net, 1, activation='sigmoid')
neural_net = tflearn.regression(neural_net, optimizer='sgd', learning_rate=2, loss='mean_square')

# Train the neural network 
model = tflearn.DNN(neural_net)
model.fit(NOR,Y_truth,n_epoch=2000,snapshot_epoch=False)

# Test final prediction
print("Testing NOR operator")
print("0 or 0:", model.predict([[0., 0. ]]))
print("0 or 1:", model.predict([[0., 1. ]]))
print("1 or 0:", model.predict([[1., 0. ]]))
print("1 or 1:", model.predict([[1., 1. ]]))

