import unittest
import numpy as np

from optimization_methods import Goldstein_Wolfe_method
from optimization_methods import optimizationProblem
from optimization_methods import rosenbrock_function, rosenbrock_function_gradient
class TestOptimization(unittest.TestCase):
    def test_function(self):
        print("\n Testing GW method")
        Wolfe_rosenbrock_function = optimizationProblem(rosenbrock_function(), rosenbrock_function_gradient())
        Wolfe = Goldstein_Wolfe_method(Wolfe_rosenbrock_function, [-0.4, 0.8])
        expected_solution = np.array([1.0, 1.0])
        
        np.testing.assert_allclose(Wolfe.run(),expected_solution, atol = 1e-1)
        
if __name__ == "__main__":
    unittest.main()