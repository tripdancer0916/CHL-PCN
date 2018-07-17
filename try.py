# coding:utf-8

import numpy as np
import cupy as cp
import os
import chainer
import PIL
import matplotlib.pyplot as plt

# Load the MNIST dataset
train, test = chainer.datasets.get_mnist()
x_train, t_train = train._datasets
x_test, t_test = test._datasets
train_size = x_train.shape[0]
batch_size = 10

x_train = cp.asarray(x_train)
x_test = cp.asarray(x_test)

t_train = cp.identity(10)[t_train.astype(int)]
t_test = cp.identity(10)[t_test.astype(int)]

weight_init_std = 0.01
hidden_units = 1000
W1 = weight_init_std * cp.random.randn(784, hidden_units)
W2 = weight_init_std * cp.random.randn(hidden_units, 10)
B2 = weight_init_std * cp.random.randn(10, hidden_units)
B1 = weight_init_std * cp.random.randn(hidden_units, 784)


def relu(x):
    return cp.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - cp.max(x, axis=0)
        y = cp.exp(x) / cp.sum(cp.exp(x), axis=0)
        return y.T

    x = x - cp.max(x)
    return cp.exp(x) / cp.sum(cp.exp(x))


def predict(x):
    h = relu(cp.dot(x, W1))
    output_ = softmax(cp.dot(h, W2))
    return output_


def accuracy(x, t):
    y = predict(x)
    y = cp.argmax(y, axis=1)
    t = cp.argmax(t, axis=1)

    accuracy = cp.sum(y == t) / float(x.shape[0])
    return accuracy


iter_per_epoch = 100
for i in range(10000):
    data_index = cp.random.choice(train_size, batch_size)
    input = x_train[data_index]
    target = t_train[data_index]
    h_forward = relu(cp.dot(input, W1))
    output = relu(cp.dot(h_forward, W2))
    h_backward = relu(cp.dot(target, B2))
    learning_rate = 0.05
    delta_W1 = learning_rate*cp.dot(input.T, (h_backward-h_forward)/batch_size)
    delta_W2 = learning_rate*cp.dot(h_forward.T, (target-output)/batch_size)
    delta_B2 = learning_rate*cp.dot(target.T, (h_forward-h_backward)/batch_size)
    W1 -= delta_W1
    W2 -= delta_W2
    B2 -= delta_B2
    if i % iter_per_epoch == 0:
        train_acc = accuracy(x_train, t_train)
        test_acc = accuracy(x_test, t_test)
        print("epoch:", int(i / iter_per_epoch), " train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
