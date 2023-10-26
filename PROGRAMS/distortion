import numpy as np
from scipy.special import comb
class DewarpingCalibrationCorrected:

    def __init__(self, degree=5):
        self.degree = degree
        self.coefficients = None
        self.q_min = None
        self.q_max = None

    @staticmethod
    def scale_to_box(data, q_min, q_max):
        return (data - q_min) / (q_max - q_min)

    @staticmethod
    def bernstein(N, k, u):
        return comb(N, k) * (1 - u) ** (N - k) * u ** k

    def construct_f_matrix(self, u):
        n_points = len(u)
        f_mat = np.zeros([n_points, (self.degree + 1) ** 3])

        for n in range(n_points):
            c = 0
            for i in range(self.degree + 1):
                for j in range(self.degree + 1):
                    for k in range(self.degree + 1):
                        f_mat[n][c] = self.bernstein(self.degree, i, u[n][0]) * \
                                      self.bernstein(self.degree, j, u[n][1]) * \
                                      self.bernstein(self.degree, k, u[n][2])
                        c += 1
        return f_mat

    def fit(self, distorted_data, ground_truth):
        self.q_min = np.min(distorted_data, axis=0)
        self.q_max = np.max(distorted_data, axis=0)

        normalized_data = DewarpingCalibrationCorrected.scale_to_box(distorted_data, self.q_min, self.q_max)
        F = self.construct_f_matrix(normalized_data)

        self.coefficients = np.linalg.lstsq(F, ground_truth, rcond=None)[0]

    def correct(self, data):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")

        normalized_data = DewarpingCalibrationCorrected.scale_to_box(data, self.q_min, self.q_max)
        F = self.construct_f_matrix(normalized_data)

        corrected_data = F @ self.coefficients
        return corrected_data


# Create an instance of the DewarpingCalibrationCorrected class, fit it to our data, and correct a sample point

###TESTING###
distorted_data = np.random.rand(10000, 3) * 10
ground_truth_data = distorted_data + np.random.randn(10000, 3) * 0.1
calibrator_corrected = DewarpingCalibrationCorrected()
sample_data = np.array([[5, 5, 5],[1,1,1]])
calibrator_corrected.fit(distorted_data, ground_truth_data)

corrected_sample = calibrator_corrected.correct(sample_data)

print(corrected_sample)
