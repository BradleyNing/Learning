import numpy as np
#import os

x = np.ma.array(np.arange(9.).reshape(3, 3), mask=[[0, 1, 0],[1, 0, 1],[0, 1, 0]])
