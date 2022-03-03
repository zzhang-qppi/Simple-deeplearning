import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(n_dimensions_f):
	#n_dimensions_f是一个包含每一层节点数的数组（包括输入层和输出层）
	L = len(n_dimensions_f)-1
	parameters = {}
	for i in range(1, L+1):
		parameters['W%d'%i] = np.random.randn(n_dimensions_f[i], n_dimensions_f[i-1]) * 0.01
		parameters['b%d'%i] = np.zeros((n_dimensions_f[i], 1))
	return parameters

def forward_propogation(A_prev, W, b, activation_func):
	if activation_func == 'sigmoid':
		Z = np.dot(W, A_prev) + b
		A = 1/(1+np.exp(-Z))
	elif activation_func == 'relu':
		Z = np.dot(W, A_prev) + b
		A = np.maximum(0, Z)
	linear_cache = (A_prev, W, b)
	activation_cache = Z
	cache = (linear_cache, activation_cache)
	return A, cache

def forward_in_total(A0, parameters):
	caches = []
	A = A0
	L = len(parameters)/2
	for i in range(1, L):
		A_prev = A
		W = parameters['W%d'%i]
		b = parameters['b%d'%i]
		A, cache = forward_propogation(A_prev, W, b, activation_func='sigmoid')
		caches.append(cache)
	A_output, cache = forward_propogation(A, 'W%d'%L, 'b%d'%L)
	caches.append(cache)
	assert A_output.shape == (1,A0.shape[1])
	assert len(caches) == L
	return A_output, caches

def cost_calculation(A_output, Y):
	m = Y.shape[1]
	cost = -np.sum(np.multiply(np.log(A_output),Y) + np.multiply(np.log(1 - A_output), 1 - Y))/m
	return cost

def sigmoid_backward(dA, Z):
	a = 1/(1+np.exp(-Z))
	dZ = dA * a * (1-a)
	return dZ

def relu_backward(dA, Z):
	dZ = np.array(dA, copy=True)
	dZ[Z <= 0] = 0
	return dZ

def linear_backward(dZ, linear_cache):
	A_prev, W, b = linear_cache
	m = A_prev.shape[1]
	dW = np.dot(A_prev.T, dZ) / m
	db = np.sum(dZ, axis=1, keepdims=True) / m
	dA_prev = np.dot(W.T, dZ)
	return dA_prev, dW, db

def linear_activation_backward(dA, linear_cache, activation_cache, activation_func):
	if activation_func == 'sigmoid':
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	elif activation_func == 'relu':
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	return dA_prev, dW, db

def backward_in_total(A_output, train_Y_f, caches):
	gradients = {}
	dA_output = - (np.divide(train_Y_f, A_output) - np.divide(1 - train_Y_f, 1 - A_output))
	dA = dA_output
	for i in range(1, len(caches)+1):
		dA = dA_prev
		linear_cache, activaiton_cache = caches[len(caches)-i]
		dA_prev, dW, db = linear_activation_backward(dA, linear_cache, activaiton_cache)
		gradients['dA%d'%(len(caches)+1-i)]=dA_prev
		gradients['dW%d'%(len(caches)+1-i)]=dW
		gradients['db%d'%(len(caches)+1-i)]=db
	return gradients

def update_parameters(parameters, gradients, l_rate_f):
	L = len(parameters)/2
	for i in range(1, L+1):
		parameters['w%d'%i] = parameters['w%d'%i] - l_rate_f * gradients['w%d'%i]
		parameters['b%d'%i] = parameters['b%d'%i] - l_rate_f * gradients['b%d'%i]
	return parameters

def multi_layers_learning(train_X_f, train_Y_f, n_dimensions_f, l_rate_f, n_iterations_f):
	parameters = initialize_parameters(n_dimensions_f)
	costs = []
	for i in range(n_iterations_f):
		A_output, caches = forward_in_total(train_X_f, parameters)
		cost = cost_calculation(A_output, train_Y_f)
		costs.append(int(cost))
		gradients = backward_in_total(A_output, train_Y_f, caches)
		parameters = update_parameters(parameters, gradients, l_rate_f)
	costs_plot, ax = plt.subplots()
	ax.plot(range(len(costs)), costs, linewidth=2.0)
	ax.set(xlable='iterations', ylable='costs', title='deep_costs_plot')
	plt.show()
	return parameters

def test(test_X_f, test_Y_f, parameters):
	A_output, caches = forward_in_total(test_X_f, parameters)
	A_output[A_output>=0.5]=1
	A_output[A_output<0.5]=1
	assert(A_output.shape == test_Y_f.shape)
	compare_result = (A_output == test_Y_f)
	accuracy = np.sum(compare_result)/test_X_f.shape[1]
	return accuracy


#parameters = multi_layers_learning(train_X, train_Y, n_dimensions, l_rate, n_iterations)
#print('模型准确度：' + test(test_X, test_Y, parameters))