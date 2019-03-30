# cosine270.py
import numpy as np

x1 = np.array([1,0])
y1 = np.array([0,-1])
xnorm1  = np.linalg.norm(x1)
ynorm1  = np.linalg.norm(y1)
xdoty1  = np.dot(x1,y1)
cosang1 = xdoty1/(xnorm1*ynorm1)
print('x1:     ',x1)
print('y1:     ',y1)
print('xnorm1: ',xnorm1)
print('ynorm1: ',ynorm1)
print('xdoty1: ',xdoty1)
print('cosang1:',cosang1)