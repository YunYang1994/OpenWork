#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : solve.py
#   Author      : YunYang1994
#   Created date: 2021-11-16 10:44:48
#   Description :
#
#================================================================

from scipy.optimize import minimize
import numpy as np

class GradientDescentLSMSolver:
    def __init__(self, A_list, b_list, learing_rate):
        self.learing_rate = learing_rate
        self.A_list = A_list
        self.b_list = b_list

    def solveGradients(self, x):
        gradients = []
        for A, b in zip(self.A_list, self.b_list):
            grad = 2 * A.T @ ( A @ x - b )
            gradients.append(grad)
        mean_grad = np.array(gradients).mean(0)
        return mean_grad

    def updateGradients(self, x, grad):
        x = x - self.learing_rate * grad
        return x

    def optimize(self, x):
        for epoch in range(1000):
            g = self.solveGradients(x)
            x = self.updateGradients(x, g)
            print(x)
        return x


A = np.arange(6).reshape(2,3)*10
true_x = np.array([1,2,3])
b = A @ true_x
A_list = [A+np.random.random(size=(2,3)) for i in range(300)] # (100, 2, 3)
b_list = [b+np.random.random(size=(2,))  for i in range(300)]  # (100, 2)

init_x = np.zeros(3)
S = GradientDescentLSMSolver(A_list, b_list, 1e-4)
pred_x = S.optimize(init_x)
print("method 1:", pred_x)

# method 2
def computeLoss(t, A, b):
    loss = np.mean(np.sum((A.dot(t) - b)**2, axis=1))
    return loss
A_list = np.array(A_list)
b_list = np.array(b_list)
t = np.zeros(3)
# result = minimize(computeLoss, t, args=(A_list, b_list), method='BFGS', options={'maxiter': 500})
result = minimize(computeLoss, t, args=(A_list, b_list), method='SLSQP', options={'maxiter': 500})
print("method 2:", result.x)
