import numpy as np

class NewtonGaussInterpolator:
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.coeffs = self._compute_divided_differences()

    def _compute_divided_differences(self):
        n = len(self.x_data)
        coeffs = np.zeros(n)
        coeffs[0] = self.y_data[0]

        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                coeffs[i] = (coeffs[i] - coeffs[i-1]) / (self.x_data[i] - self.x_data[i-j])

        return coeffs

    def interpolate(self, x):
        n = len(self.coeffs)
        result = self.coeffs[-1]
        for i in range(n-2, -1, -1):
            result = result * (x - self.x_data[i]) + self.coeffs[i]
        return result

    def __call__(self, x):
        if x < self.x_data[0] or x > self.x_data[-1]:
            return np.nan
        return self.interpolate(x)