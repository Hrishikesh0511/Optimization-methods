import os

from prettytable import PrettyTable
import numpy as np
import numpy.typing as npt

from algos import conjugate_descent, sr1, dfp, bfgs
from functions import (
    trid_function,
    trid_function_derivative,
    three_hump_camel_function,
    three_hump_camel_function_derivative,
    rosenbrock_function,
    rosenbrock_function_derivative,
    styblinski_tang_function,
    styblinski_tang_function_derivative,
    func_1,
    func_1_derivative,
    matyas_function,
    matyas_function_derivative,
    hyperEllipsoid_function,
    hyperEllipsoid_derivative,
)

test_cases = [
    [
        trid_function,
        trid_function_derivative,
        np.asarray([-2.0, -2]),
    ],
    [
        trid_function,
        trid_function_derivative,
        np.asarray([-2.0, -2]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        np.asarray([-2.0, 1]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        np.asarray([2.0, -1]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        np.asarray([-2.0, -1]),
    ],
    [
        three_hump_camel_function,
        three_hump_camel_function_derivative,
        np.asarray([2.0, 1]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        np.asarray([2.0, 2, 2, -2]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        np.asarray([2.0, -2, -2, 2]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        np.asarray([-2.0, 2, 2, 2]),
    ],
    [
        rosenbrock_function,
        rosenbrock_function_derivative,
        np.asarray([3.0, 3, 3, 3]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        np.asarray([0.0, 0, 0, 0]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        np.asarray([3.0, 3, 3, 3]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        np.asarray([-3.0, -3, -3, -3]),
    ],
    [
        styblinski_tang_function,
        styblinski_tang_function_derivative,
        np.asarray([3.0, -3, 3, -3]),
    ],
    [
        func_1,
        func_1_derivative,
        np.asarray([3.0, 3]),
    ],
    [
        func_1,
        func_1_derivative,
        np.asarray([-0.5, 0.5]),
    ],
    [
        func_1,
        func_1_derivative,
        np.asarray([-3.5, 0.5]),
    ],
    [matyas_function, matyas_function_derivative, np.asarray([2.0, -2])],
    [matyas_function, matyas_function_derivative, np.asarray([1, 10.0])],
    [hyperEllipsoid_function, hyperEllipsoid_derivative, np.asarray([-3, 3, 2.0])],
    [
        hyperEllipsoid_function,
        hyperEllipsoid_derivative,
        np.asarray([10, -10, 15, 15, -20, 11, 312.0]),
    ],
]
i = 3
ans = conjugate_descent(test_cases[i][2], test_cases[i][0], test_cases[i][1],"Hestenes-Stiefel")
print(ans)
