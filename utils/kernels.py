import numpy as np



class Kernel:

    def __init__(self, kernel_str) -> None:
        assert kernel_str in [
            "gaussian", 
            "uniform", 
            "epanechnikov", 
            "quadratic", 
            "triangular", 
            "one-sided epanechnikov (R)",
            "one-sided epanechnikov (L)"
            ], f"{kernel_str}"
        self.kernel_str = kernel_str

    def __call__(self, x):
        if self.kernel_str == "gaussian":
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-(x ** 2) / 2)
        elif self.kernel_str == "uniform":
            return 1 / 2 * (np.abs(x) <= 1) 
        elif self.kernel_str == "epanechnikov":
            return (1 - x ** 2) * 3 / 4 * (np.abs(x) <= 1)
        elif self.kernel_str == "quadratic":
            return 15 / 16 * (1 - x ** 2) ** 2 * (np.abs(x) <= 1)
        elif self.kernel_str == "triangular":
            return (1 - np.abs(x)) * (np.abs(x) <= 1)
        elif self.kernel_str == "one-sided epanechnikov (R)":
            return (1 - x ** 2) * 3 / 2 * (x > 0) * (x < 1)
        elif self.kernel_str == "one-sided epanechnikov (L)":
            return (1 - x ** 2) * 3 / 2 * (x > -1) * (x < 0)
    