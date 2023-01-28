from utils import *
import CSsolver
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class RD_SI_sampling(object):
    def __init__(
            self,
            sensing_mat,
            transform_mat,
            si_kernel,
            integration_oversampling,
            input_signal_resolution,
            si_T,
            sparsity
    ):
        self.sensing_mat = sensing_mat
        self.transform_mat = transform_mat
        self.M = sensing_mat.shape[0]
        self.N = sensing_mat.shape[1]
        self.zi = np.repeat(sensing_mat, integration_oversampling, axis=1)
        self.zi = np.append(self.zi, np.transpose([self.zi[:,-1]]), axis = 1)
        print(self.zi.shape)
        self.si_kernel = si_kernel
        self.integration_oversampling = integration_oversampling
        self.input_T = input_signal_resolution
        self.si_T = si_T
        self.down_sample = self.si_T / self.input_T
        self.integration_range = \
            np.arange(0, self.N+0.00001, 1/integration_oversampling)
        self.CSsolver = CSsolver.SpamsOmpSolver(sensing_mat @ transform_mat, sparsity)

    def measure_slice(self, x_slice):
        """

        :param x_slice: must include the last point so that
        :return:
        """
        ts = np.arange(0, N + 1/down_sample, 1/down_sample)
        x_interpolator = sp.interpolate.interp1d(ts, x_slice, fill_value="extrapolate")
        x_slice_interp = x_interpolator(self.integration_range)
        xz = self.zi * x_slice_interp
        return sp.integrate.trapezoid(xz, self.integration_range)

    def reconstruct_slice(self, measurments, method="omp", parameter=0):
        if method == "omp":
            x_likely = self.CSsolver.solve(measurments, parameter)
        elif method == "lasso":
            x_likely = self.CSsolver.solve_lasso(measurments, parameter)
        ck_likely = self.transform_mat @ x_likely
        return ck_likely


# Globals
track_length = 30
down_sample = 8
si_spline_degree = 3
M = 500
N = 1024
integration_oversampling = 100

# Test data prep
fs, x = get_test_data()
x = x[:fs*track_length]
x = x/np.max(np.abs(x))
T_orig = 1/fs

# Resampled SI signal at 24khz
T = T_orig * down_sample
srange = np.arange(-6, 6+T_orig/T, 1/down_sample)
a_si = sp.signal.bspline(srange, 0)
new_x = project_into_si_signal(x, T_orig, a_si, T)

sensing_mat = get_random_pm1_matrix(M, N)
transform_mat = np.transpose(sp.fft.idct(np.identity(1024)))
rd_sampler = RD_SI_sampling(sensing_mat, transform_mat, a_si, integration_oversampling, T_orig, T, 170)

x_w = x[fs*18:fs*18 + N*down_sample+1]
yi = rd_sampler.measure_slice(x_w)
sparcities_array = np.arange(1, 201, 1)
ck_likely_omp = np.array([rd_sampler.reconstruct_slice(yi, "omp", L) for L in sparcities_array])

lambda1_array = np.exp2(-np.arange(0, 10, 0.05))
ck_likely_lasso = np.array([rd_sampler.reconstruct_slice(yi, "lasso", L) for L in lambda1_array])


# utility
def db(x):
    return 20*np.log10(np.abs(x)+0.0000000001)

# verifing yi is as we wanted
integration_oversampling2 = 10000
integration_range = np.arange(0, N*down_sample, down_sample/integration_oversampling2)
x_w_interp = sp.interpolate.interp1d(np.arange(N*down_sample+1), x_w)(integration_range)
s = np.ones(integration_oversampling2)
xcs = sp.signal.convolve(x_w_interp, s[::-1], mode="same")/integration_oversampling2
ck = xcs[np.arange(integration_oversampling2//2, N*integration_oversampling2, integration_oversampling2)]
print(f"yi calculation accuracy: {db(np.std(sensing_mat @ ck - yi)/np.std(yi))}")

ck_prediction_omp = db(np.std(ck_likely_omp - ck, axis=1)/np.std(ck))
plt.plot(ck_prediction_omp, color="blue")

yi_prediction_omp = db(np.std(np.transpose(sensing_mat @ np.transpose(ck_likely_omp)) - yi, axis=1)/np.std(yi))
plt.plot(yi_prediction_omp, color="lightblue")

ck_prediction_lasso = db(np.std(ck_likely_lasso - ck, axis=1)/np.std(ck))
plt.plot(ck_prediction_lasso, color="red")

yi_prediction_lasso = db(np.std(np.transpose(sensing_mat @ np.transpose(ck_likely_lasso)) - yi, axis=1)/np.std(yi))
plt.plot(yi_prediction_lasso, color="orange")

plt.show()
