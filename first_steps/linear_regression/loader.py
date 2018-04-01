import numpy as np


def loadData(location):
    """Loads a features matrix and a Y vector from a given path to a text file

    location -- A valid location to a text file. The text file must conform to the following format:
    feature1 <2 spaces> feature2 <2 spaces> ... label
    MAKE SURE YOU ADD ONES COLUMN TO YOUR MATRIX

    :return A NumPy matrix of the features of the dimensions: <# Of Data sets> X <# Of features>, and a Y vector
    """
    content = open(location, "r") # Just read
    data = []
    y = []
    for line in content:
        splitted = line.split("  ")
        features = splitted[0:len(splitted)-1]
        y.append(float(splitted[len(splitted) - 1]))

        data_set = []
        for f in features:
            data_set.append(float(f))
        data.append(data_set)

    return np.matrix(data, dtype=np.float64),np.matrix(y, dtype=np.float64)


def normalize_prices(D, index):
    i = 0
    while i < len(D):
        D[i, index] /= 1000
        i += 1
    return D


def add_ones_column(D):
    return np.append(np.ones((len(D), 1)), D, axis=1)


def tune_matrix(D):
    D = normalize_prices(D, 0)
    D = add_ones_column(D)
    return D


def test():
    matrix, y = loadData("../smartphone.txt")
    print(str(tune_matrix(matrix)))


# test()