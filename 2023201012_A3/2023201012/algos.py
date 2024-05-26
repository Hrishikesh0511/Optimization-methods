from typing import Callable, Literal

import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np
def projected_gd(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    point: npt.NDArray[np.float64],
    constraint_type: Literal["linear", "l_2"],
    constraints: npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], np.float64]
) -> npt.NDArray[np.float64]:
    
    x0 = point
    eps = 1e-6
    x_k = x0
    x_k1 = x0    
    k = 0

    # function for finding Pc
    def Pc(point: npt.NDArray[np.float64]):
        if constraint_type == "linear":
            li:npt.NDArray = constraints[0]
            ui:npt.NDArray = constraints[1]
            if (point <= li).any():
                return li
            elif (point >= ui).any():
                return ui
            else:
                return point
        else:
            c,r = constraints
            nf = np.linalg.norm(point - c )
            return r*(point - c)/max(r,nf) + c
    
    def Gm(point: npt.NDArray[np.float64],M):
        pt = Pc(point - M*d_f(point))
        return (1/(M))*(point - pt)
    
    def backtrack(point: npt.NDArray[np.float64]):
        beta = 0.9
        s = 1
        t_k = s
        # print ("Constraint")
        k = 0
        alpha = 0.001
        while((f(point)-f(Pc(point - t_k*d_f(point))) < alpha*t_k*(np.linalg.norm(Gm(point,t_k),ord = 2)**2)) and k < 1000):
            t_k = beta*t_k
            k+=1
        return t_k
        # while((abs(f(point)-f(Pc(point - t_k*d_f(point))) - t_k*(np.linalg.norm(Gm(point,t_k),ord = 2)**2))) <= 1e-3 and k < 1000):
        #     t_k = beta*t_k
        #     k+=1
        # return t_k
    
    while ( k < 1000):
        x_k = x_k1
        tk = backtrack(x_k)
        x_k1 = Pc(x_k - tk*d_f(x_k))
        if(np.linalg.norm(x_k1 - x_k) <= eps):
            break
        k+=1
    return x_k1

def dual_ascent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    c: list[Callable[[npt.NDArray[np.float64]], np.float64 | float]],
    d_c: list[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    initial_point: npt.NDArray[np.float64],
):
    alpha = 1e-3
    k = 0
    x_k = initial_point
    lambdas = np.ones(len(d_c))
    max_iter = 1e5
    while k < max_iter:
        l_x = d_f(x_k) + sum(np.array([lambda_i*d_c_i(x_k) for lambda_i,d_c_i in zip(lambdas,d_c)]))
        x_k = (x_k - alpha*l_x)
        l_lambda = np.array([c_i(x_k) for c_i in c])
        lambdas = lambdas +  alpha*l_lambda
        lambdas = np.array([max(0,i) for i in lambdas])
        k = k + 1

    return (x_k,lambdas)

