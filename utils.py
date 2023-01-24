from scipy.io import wavfile
import numpy as np


def get_test_data(channel="r"):
    fs, data = wavfile.read("test_data/Ilg_Fuoco_1.wav")
    data_r, data_l = np.transpose(data)
    if channel == "r":
        return fs, data_r
    elif channel == "l":
        return fs, data_l
    else:
        raise Exception("Channel must be either 'l' or 'r'")


def get_random_pm1_matrix(rows, cols):
    """
    Builds a random +-1 matrix in numpy array format with shape (rows, cols)
    """
    return (-1) ** np.random.randint(0, 2, (rows, cols))


def get_random_sparse_vector(n, d, minv=-1, maxv=1):
    """
    generates a random d-sparse array of size n with nnz values in the range (minv, maxv)
    :param n: dimension
    :param d: number of nnz
    :param minv: lower bound of possible nnz values
    :param maxv: upper bound of possible nnz values
    """
    x = np.zeros(n)
    x[np.random.choice(np.arange(0, n), d, replace=False)] = (np.random.random(d)*(maxv - minv) + minv)
    return x


