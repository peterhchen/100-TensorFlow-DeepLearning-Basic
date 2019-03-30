from sklearn import tree # Filename: predict-sklearn.py
# step 1: collect training data
features = [[130,1], [140,1], [150,0], [170,0]]
labels   = [0, 0, 1, 1]
# step 2: train the classifier via a decision tree
clf = tree.DecisionTreeClassifier()

# "fit" learning algorithm ("find patterns in data")
clf = clf.fit(features, labels)
# step 3: make a prediction
print ('a fruit of weight 160 =',clf.predict([[160,0]]), ' where 0 is an apple and 1 is an orange')