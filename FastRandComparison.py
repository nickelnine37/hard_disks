import fastrand
import time
import numpy as np
import matplotlib.pyplot as plt


# Time tests

# Fastrand32 float
t0 = time.time()
for i in range(10000000):
    my_random_float = fastrand.pcg32() / 2147483647
print('fastrand32 float: ', time.time() - t0, 's')

# Numpy float
t0 = time.time()
for i in range(10000000):
    my_random_float = 2 * (np.random.rand() - 0.5)
print('numpy float: ', time.time() - t0, 's')

# Fastrand int
t0 = time.time()
for i in range(10000000):
    my_random_int = fastrand.pcg32bounded(250)
print('fastrand int: ', time.time() - t0, 's')

# Numpy int
t0 = time.time()
for i in range(10000000):
    my_random_int = np.random.randint(0, 250)
print('numpy int: ', time.time() - t0, 's')


# Uniformity check

fr_data = np.zeros(100000)
np_data = np.zeros(100000)

for i in range(len(fr_data)):
    fr_data[i] = fastrand.pcg32() / 2147483647
    np_data[i] = 2 * (np.random.rand() - 0.5)

plt.figure()
plt.subplot(2, 1, 1)
n1, bins1, patches1 = plt.hist(fr_data, 50, normed=1, facecolor='y', alpha=0.75)
plt.title('Fastrand')

plt.subplot(2, 1, 2)
n2, bins2, patches2 = plt.hist(np_data, 50, normed=1, facecolor='g', alpha=0.75)
plt.title('Numpy')
plt.tight_layout()
plt.show()
