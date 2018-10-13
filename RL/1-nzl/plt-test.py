import numpy as np
import math
import matplotlib.pyplot as plt


lbd = 3
x = []
y = []
for i in range(15):
	yi = np.power(lbd, i)*np.exp(-lbd)/np.math.factorial(i)
	x.append(i)
	y.append(yi)
	

print(x, y)
print(np.sum(y))
plt.plot(x, y)
plt.show()
