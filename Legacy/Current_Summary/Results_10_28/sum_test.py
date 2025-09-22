import numpy as np
import matplotlib.pyplot as plt


n_terms=2000

# Define the sequence {s_n} starting at 1, with each term being one-fifth of the previous term


# Calculate the sum of the sequence
fig=plt.figure(figsize=(5,5))

def ratio_sum(param):   
    sum_s_n = np.zeros(n_terms)
    for i in range(1, n_terms):
        sum_s_n[i] = np.sum(np.array([1 * (1/param)**j for j in range(0,i+1)]))

    plt.plot(np.linspace(1,n_terms, n_terms)[1:],sum_s_n[1:],label='param='+str(param))

    print(sum_s_n)

for param in [1.1,1.2,1.3,2,3,4]:
    ratio_sum(param)

plt.legend()
plt.show()

