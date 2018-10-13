import numpy as np

mylist = [1,2,3]
print mylist

myarray = np.array(mylist)
#myarray = [1 2 3]
print myarray

# import matplotlib.pyplot as plt
# from matplotlib.table import Table

# fig, ax = plt.subplots()
# ax.set_axis_off()
# tb = Table(ax, bbox=[0, 0, 1, 1])

# nrows, ncols = (5, 5)
# width, height = 1.0 / ncols, 1.0 / nrows
# i = 0
# j = 0 
# for i in range(5):
# 	for j in range(5):
# 		tb.add_cell(i, j, width, height, text=(i+1)*(j+1), loc='center')

# ax.add_table(tb)
# plt.show()