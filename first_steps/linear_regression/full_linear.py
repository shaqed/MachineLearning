# This is a full implementation of the Gradient Descent for multi-variable linear regression
import loader
import numpy as np


def load_data():
    matrix, y_vector = loader.loadData("../smartphone.txt")
    matrix = loader.tune_matrix(matrix)
    y_vector = np.transpose(y_vector)
    return matrix, y_vector


def run(debug = True, alpha = 0.1, max_steps = 3000):
    matrix, y_vector = load_data()

    num_of_features = matrix.shape[1]
    num_of_data_sets = matrix.shape[0]

    # For starter, the hypothesis (the coefficients of the linear functions) is zero
    # That means, every single coefficient is zero
    hypothesis = np.zeros((num_of_features,1))

    if debug:
        print("Starting training algorithm:")
        print("Initial Data:\n-------")
        print("Matrix: " + str(matrix))
        print("Y_Vector: " + str(y_vector))
        print("# of features (m): " + str(num_of_features))
        print("# of data sets (n): " + str(num_of_data_sets))
        print("-------\nStarting loops (max: " + str(max_steps) + ")")

    min_cost = np.inf
    for i in range(max_steps):

        # Predictions = T0 + X1*T1 + X2*T2 + X3*T3 + ... = Y'
        # Y' is a vector with the same length of Y
        # And it holds the predicted values of data sets based on the hypothesis
        # For example, for data set #i you can look up at Y'(i)
        # And compare that against the true value of it at Y(i)
        predictions = np.matmul(matrix, hypothesis)
        # print("Prediction vector: " + str(predictions))

        # Cost - The regular cost function, the average of all of the squared
        # differences between each Y'(i) and Y(i)
        # To compute, calculate the Y' and subtract Y from it
        # Square the vector (square each item in the new vector)
        # Divide it by the number of data sets to get the average
        cost = np.sum(np.square(predictions - y_vector)) / (1*num_of_data_sets)

        # A vector the same size as the amount of data sets you have
        # Each i'th element contains the difference between the true value of i, Y(i)
        # And the predicted value of i, Y'(i)
        differences_vector = (predictions - y_vector)

        # Gradient vector, to compute it - for each i'th element in the vector we need to:
        # Sum the differences between Y'(i) and Y(i)
        # Multiply that by Xi of that i'th data set
        # Multiply again by 2/m
        gradient_vector = 2*(np.matmul(np.transpose(matrix), differences_vector))/ (num_of_data_sets)

        # Update the hypothesis by the following function,
        # This is where we see the 'alpha' parameter taking place
        hypothesis = hypothesis - alpha * gradient_vector

        if debug:
            print(str(i) + ": " + str(np.transpose(hypothesis)) + ". Cost: " + str(cost))

        if cost < min_cost:
            min_cost = cost
        else:
            return min_cost, i, hypothesis
    return min_cost, max_steps, hypothesis


def best_alpha():
    """You may use this function to help getting the best alpha, of course this needs
    an initial guess as well"""
    lowest_steps = np.inf
    lowest_cost = np.inf

    alpha = 0.1
    jump_by = alpha / 2.0
    for i in range(100):
        cost, steps_took, hypo = run(debug=False, alpha=alpha)
        if steps_took < lowest_steps and cost <= lowest_cost:
            print("a: " + str(alpha) + " was good [cost=" + str(cost) + ", steps=" + str(steps_took) + "]")
            lowest_steps = steps_took
            lowest_cost = cost
            alpha += jump_by # increase alpha by the interval
        else:
            # That alpha increase was a bad idea, go back
            print("a: " + str(alpha) + "\t was bad [cost=" + str(cost) + ", steps=" + str(steps_took) + "]")
            alpha -= jump_by
            # divide by half the interval
            jump_by = (jump_by) / 2.0
            alpha += jump_by


# Run the algorithm
ans = run(debug=True, alpha=0.05)

