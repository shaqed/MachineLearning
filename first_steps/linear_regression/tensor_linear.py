import tensorflow as tf
import numpy as np
import loader


def load_data():
    matrix, y_vector = loader.loadData("../smartphone.txt")
    matrix = loader.tune_matrix(matrix)
    y_vector = np.transpose(y_vector)
    return matrix, y_vector


def run_without_matrix(debug = True, alpha = 0.1, max_steps = 3000):
    """Run with a fixed linear regression of 2 features only"""
    matrix, y_vector = load_data()

    # initialize data
    x1 = (matrix[:,1])
    x2 = (matrix[:,2])
    y_vector = (y_vector)

    # Hypothesis
    hypo_1 = tf.Variable(0, dtype=tf.float64)
    hypo_2 = tf.Variable(0, dtype=tf.float64)
    hypo_3 = tf.Variable(0, dtype=tf.float64)


    # Gradient Descent
    predictions = hypo_1 + hypo_2 * x1 + hypo_3 * x2
    cost = tf.reduce_mean(tf.square(predictions - y_vector))
    # train = tf.train.GradientDescentOptimizer(tf.Variable(alpha)).minimize(cost)

    gradient = tf.train.GradientDescentOptimizer(tf.Variable(alpha))
    grad_vector = gradient.compute_gradients(cost)
    grad_applied = gradient.apply_gradients(grad_vector)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(max_steps):
            theta = sess.run([grad_applied, hypo_1, hypo_2, hypo_3, cost])
            print("Answer: " + str(theta))


def run_with_matrix(debug = True, alpha = 0.1, max_steps = 3000):
    """This function can take up to as many features you want"""

    # Load the data
    matrix, y_vector = load_data()

    # Setup the data
    num_of_features = matrix.shape[1] # Number of features
    matrix = tf.constant(matrix, dtype=tf.float64) # Data matrix
    y_vector = tf.constant(y_vector, dtype=tf.float64) # Y vector

    # Initialize the hypothesis with zeros as a zero vector
    hypothesis = tf.Variable(tf.zeros([num_of_features, 1], dtype=tf.float64))

    # Predictions of everything can be produced by multiplying the matrix with the hypothesis vector
    # This will yield a vector with the same dimensions as Y
    # And we will be able the compare between the two
    predictions = tf.matmul(matrix, hypothesis)

    # The cost function, we would like this function to yield the smallest value as possible
    cost = tf.reduce_mean(tf.square(predictions - y_vector))

    # This is where TensorFlow really kicks in
    # We use the GradientDescentOptimizer with a given alpha
    # And on that object we call the minimize() function by supplying the cost we wish to "minimize"
    # This line of code takes care of the gradient update and hypothesis for us
    train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

    # TensorFlow stuff, intialize all variables
    init = tf.global_variables_initializer()

    best_cost = np.inf

    # Start a session, we will use the 'sess' object here to run the commands decalred above
    with tf.Session() as sess:
        sess.run(init)  # Initialize variables

        for i in range(max_steps):

            # Get the result of the gradient, the hypothesis and the cost by running them
            # In practice, the running of the 'train' is the one doing the updates
            # We run the 'hypothesis' and the 'cost' just so that they will come back
            # From the function and we will be able to print the values
            grad, hypo, current_cost = sess.run([train, hypothesis, cost])

            # Print
            if debug:
                print(str(i) + ": " + str(np.transpose(hypo)) + ". Cost: " + str(current_cost))


            if current_cost < best_cost:
                best_cost = current_cost
            else:
                return best_cost, i, hypo
        return best_cost, max_steps, hypo



run_with_matrix(alpha=0.05)
