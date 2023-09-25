import numpy as np

np.set_printoptions(precision=4, suppress=True)

x1 = 3.0
x2 = -3.14
w1 = 2.1
w2 = 2.6
bias = 0.2
eta = 1.0
d = 0.798
num_iterations = 50


def activation_func(A):
    return 1.0 / (1 + np.exp(-1 * A))


"""
Activity Function sums the products of the weights
and inputs and feeds this sum into the Activation Function 
"""
print(f"Weights w1: {w1} , w2: {w2}")
activity = w1 * x1 + w2 * x2
print(f"Activity value: {activity}")
print(f"Activation value: {activation_func(activity)}")

"""
This part of the program adjusts the 
weights based on the output of the activity function
which is stored in our variable y.
"""
for i in range(0, num_iterations):
    A = w1 * x1 + w2 * x2 + bias

    # Activity function output
    y = activation_func(A)

    # adjusted error value based on
    # desired value and network output
    delta = (d - y) * y * (1 - y)

    # Adjusted weights to be fed back into
    # the Activity Function
    w1 = w1 + eta * delta * x1
    w2 = w2 + eta * delta * x2

    # Adjusted bias to be fed back into
    # the Activity Function
    bias = bias + eta * delta * 1
    print(y)
