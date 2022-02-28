from tkinter import N, W, Y
import numpy as np
from dnn_utils import sigmoid_backward
import lr_utils

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()
hyper_para = {'l_rate':0.01, 'depth':10, 'nodes':[], 'activation_func':'sigmoid', 'output_func':'sigmoid'}

assert(len(hyper_para['nodes']) == hyper_para['depth'])

def initialize(nodes, depth):
	parameters = {}
	assert(depth == len(nodes))
	for i in range(1, depth+1):
		parameters['w%d'%i] = np.random.randn(nodes[i], nodes[i-1])
		parameters['b%d'%i] = np.random.randn(nodes[i], train_set_x_orig.shape[0])
	return parameters

def linear_forward(A, W, b):
	Z = np.dot(W, A) + b
	cache = [A, W, b]
	return Z
#	return Z, cache

def activation_forward(A_prev, W, b, activation_func):
	if activation_func == 'sigmoid':
#		Z, linear_cache = linear_forward(A_prev, W, b)
		Z = linear_forward(A_prev, W, b)
		A = 1/(1+np.exp(-Z))
	elif activation_func == 'relu':
		Z = linear_forward(A_prev, W, b)
		A = np.maximum(0, Z)
#	elif activation_func == 'tanh':
	return A

def forward_prop(A, parameters, depth, activation_func, output_func):
	for i in range(1, depth):
		A = activation_forward(A, parameters['w%d'%i], parameters['b%d'%i], activation_func)
	A_output = activation_forward(A, parameters['w%d'%depth], parameters['b%d'%depth], output_func)
	return A_output

def cost(A, Y):
	m = Y.shape[1]
	cost = -np.sum(np.multiply(np.log(A),Y) + np.multiply(np.log(1 - A), 1 - Y))/m
#	cost = np.squeeze(cost)
	return cost

def activation_backward(dA, Z, activation_func):
	if activation_func == 'sigmoid':
		s = 1/(1+np.exp(-Z))
		dZ = dA * s * (1-s)
	elif activation_func == 'relu':
#		Z[Z>0] = 1
#		dZ = Z
		dZ = np.array(dA, copy=True)
		dZ[Z <= 0] = 0
	return dZ
#	elif activation_func = 'tanh':

def linear_backward(dZ, A_prev, W):
	dA_prev = np.dot(W.T, dZ)
	m = A_prev.shape[1]
	dW = np.dot(dZ, A_prev.T)/m
	db = np.sum(dZ, axis=1, keepdims=True)/m
	return dA_prev, dW, db

def linear_activation_backward(dA, A_prev, Z, activation_func):
	dZ = activation_backward(dA, Z, activation_func)
	dA_prev, dW, db = linear_backward(dZ, A_prev, W)
	return dA_prev, dW, db


def gradient_descent(W, dW, b, db, l_rate):
	W, b = W - l_rate*dW, b - l_rate*db
	return W, b

if __name__ == '__main__':

