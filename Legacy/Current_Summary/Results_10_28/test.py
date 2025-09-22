import numpy as np
import matplotlib.pyplot as plt
# Define the sequence length
n_terms = 100000000  # Number of terms to calculate the mean for

# Define the sequence {s_n} as ln(n) for n > 1
s_n = np.array([np.log(n) for n in range(1, n_terms + 1)])

# Calculate the arithmetic means σ_n for the first n terms
sigma_n = np.cumsum(s_n) / np.arange(1, n_terms + 1)
# Display the first 10 values of the arithmetic mean sequence
plt.plot(np.linspace(1,n_terms, n_terms),sigma_n)
plt.grid('on')

plt.show()


