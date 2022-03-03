import numpy
import lr_utils
import deepNN
import shallowNN

train_set_x , train_set_y , test_X , test_Y , classes = lr_utils.load_dataset()
train_X = train_set_x.reshape(train_set_x.shape[0], -1).T / 255
train_Y = train_set_y.reshape(train_set_y.shape[0], -1).T / 255

deep_dimensions = [12288, 10, 6, 5, 3, 1]
deep_parameters = deepNN.multi_layers_learning(train_X, train_Y, deep_dimensions, 0.001, 3000)

shallow_dimensions = [12288, 8, 1]
shallow_parameters = shallowNN.two_layers_learning(train_X, train_Y, shallow_dimensions, 0.001, 3000)

deep_accuracy = deepNN.test(test_X, test_Y, deep_parameters)
shallow_accuracy = deepNN.test(test_X, test_Y, shallow_parameters)
print('两层神经网络准确度：' + str(shallow_accuracy))
print('深层神经网络准确度：' + str(deep_accuracy))