import numpy as np

var=np.zeros(500,dtype=np.ndarray)
for h in range(500):
    for i in range(10):
        temp_main=np.zeros((10,10),dtype=np.ndarray)
        for j in range(10):
            if i%2==0:
                temp=np.zeros((10,10))
            else:
                temp=np.ones((10,10))   
            temp_main[i,j]=temp
    var[h]=temp_main

print(var)