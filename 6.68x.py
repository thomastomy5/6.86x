import numpy as np

def neural_network(inputs, weights):
    p = np.dot(inputs.T, weights)
    a = np.tanh(p)
    return np.sum(a, keepdims=True)


# Weights = np.array([[0.98827571], [0.30870752]])
# Input = np.array([[0.74764776], [0.62642958]])

Weights = np.array([[1], [2]])
Input = np.array([[3], [4]])

print(neural_network(Input,Weights))

print('hu')

if 'bob' in str