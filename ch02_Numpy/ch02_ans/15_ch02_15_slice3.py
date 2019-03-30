import numpy as np
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print('a:  ',a)
d1 = np.asanyarray(a[0:1])
print('d1: ',d1)
d2 = np.asanyarray(a[1:3])
print('d2: ',d2)
d3 = np.asanyarray(a[1:])
print('d3: ',d3)
d4 = np.asanyarray(a[:-1])
print('d4: ',d4)
d5 = np.asanyarray(a[-1])
print('d5: ',d5)