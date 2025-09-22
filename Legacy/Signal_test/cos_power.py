import matplotlib.pyplot as plt
import numpy as np

#the point of this excercise is to show that in the limit of large n, for odd n, the x and 3x components of the fourier decomposition become 
#unboundedly close. This causes unreliability of the fourier decomposition method to accurately resolve the true signal frequency.


#what if you search for delta functions, instead of the heart rate function?
#try using more colors 
#find when the 
#collapse the signal somehow from one perspective over intervals so that a 'delta function' is found

def func(n, xdat):
    return (np.cos(xdat))**n


xdat = np.linspace(0, 4*np.pi, 500)


cmap = plt.get_cmap('plasma')

for n in range(1, 20):
    ydat = func(n, xdat)
    plt.plot(xdat, ydat, label=f'n={n}', color = cmap(n/20))

plt.grid('on')
plt.show()
