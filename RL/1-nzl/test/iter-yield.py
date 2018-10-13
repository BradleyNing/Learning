import numpy as np
import os

policy = np.ones([16, 4]) / 4
for a, action_prob in enumerate(policy[0]):
	print a, action_prob

f = open('temp.txt')
print len([word for line in f for word in line.split()])

f.seek(0)
print sum([len(word) for line in f for word in line.split()])

f.seek(0)
print sum(len(word) for line in f for word in line.split())

rows = [1, 2, 3, 17]
def cols(): # example of simple generator
	yield 56
	yield 2
	yield 1

x_product_pairs = ((i, j) for i in rows for j in cols())
for pair in x_product_pairs:
	print pair
