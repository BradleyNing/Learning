from dnn_app_nzl_utils import *
import numpy as np

logits = np.array([0.2, 0.4, 0.7, 0.9])
labels = np.array([0, 0, 1, 1])
cost = sigmoid_cross_entropy_with_logits(logits, labels)
print ("cost = " + str(cost))

temp = sigmoid(logits)
print temp

cost = np.array([1.0053872,  1.0366408,  0.41385433, 0.39956617])
print (cost.sum())

AL = sigmoid(logits)
Y = labels.reshape(1,len(labels))
cost = compute_cost(AL, Y)
print (cost)

# import time
# import numpy as np
# #import h5py
# import matplotlib.pyplot as plt
# #import scipy
# #from PIL import Image
# #from scipy import ndimage
# #from dnn_app_utils_v2 import *
# from dnn_app_nzl_utils import *
# from opt_utils import load_dataset

# #train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# tic = time.process_time()
# # Reshape the training and test examples 
# # train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
# # test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# # # Standardize data to have feature values between 0 and 1.
# # train_x = train_x_flatten/255.
# # test_x = test_x_flatten/255.
# #print train_x.shape

# ### CONSTANTS DEFINING THE MODEL ####

# # n_x = 12288     # num_px * num_px * 3
# # layers_dims = (12288, 7, 1) # n_x, n_h, n_y
# #parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 3000, print_cost = True)

# #layers_dims = [12288, 20, 7, 5, 1]
# #parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 3000, print_cost = True)

# #pred_train = predict(train_x, train_y, parameters)
# #pred_test = predict(test_x, test_y, parameters)
# #print (pred_train, pred_test)
# train_x, train_y = load_dataset()
# plt.figure()
# layers_dims = [train_x.shape[0], 5, 2, 1]
# # by adm optimizer
# # parameters = model(train_x, train_y, layers_dims, optimizer = "adam", 
# # 				   learning_rate = 0.0007, mini_batch_size = 64, 
# # 				   num_epochs = 10000)

# # parameters = model(train_x, train_y, layers_dims, optimizer = "momentum", 
# # 				   learning_rate = 0.0007, mini_batch_size = 64, 
# # 				   num_epochs = 10000)

# parameters = model(train_x, train_y, layers_dims, optimizer = "gd", 
# 				   learning_rate = 0.0007, mini_batch_size = 64, 
# 				   num_epochs = 10000)

# pred_train = predict(train_x, train_y, parameters)
# #print (pred_train)

# # Plot decision boundary
# plt.figure()
# plt.title("Model with Adam optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5,2.5])
# axes.set_ylim([-1,1.5])
# yt = train_y.ravel()
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, yt)

# toc = time.process_time()
# print ('running duration: ' + str(toc-tic) + 's')
