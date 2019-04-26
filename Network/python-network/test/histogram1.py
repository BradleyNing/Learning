import random
import matplotlib.pyplot as plt

uniSamples = [random.random() for i in range(1000)]
print(uniSamples[0:10])  #Take a look at the first 10
fig, axis = plt.subplots()
axis.hist(uniSamples, bins=100, density=True)
axis.set_title("Histogram for uniform random number generator")
axis.set_xlabel("x")
axis.set_ylabel("normalized frequency of occurrence")
#fig.savefig("UniformHistogram.png")
plt.show()
