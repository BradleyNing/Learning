import numpy as np

def sigmoid(x):
	return (1.0 / (1.0 + np.exp(-x)))

def softmax(array):
	size = len(array)
	result = np.zeros(size)
	for i in range(size):
		result[i] = np.exp(array[i]) / np.exp(array).sum()

	return result

temp = [-0.04781715, -0.18746823]
print(softmax(temp))