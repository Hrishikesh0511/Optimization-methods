from typing import Callable, Literal

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np

def plot_graphs(condition, fx, grad_fx, f, initial_point, x):
    plt.plot(fx)
    plt.xlabel('number of iterations')
    plt.ylabel('function value')
    plt.title('f(x) vs iters')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(initial_point)}_{condition}_vals.png")
    plt.close()
    
    plt.plot([np.linalg.norm(grad_f) for grad_f in grad_fx])
    plt.xlabel('number of iterations')
    plt.ylabel('gradient norm')
    plt.title('|f\'(x)| vs iters')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(initial_point)}_{condition}_grad.png")
    plt.close()
    
    if x[0].shape[0] == 2:
        x_tmp = np.arange(-5, 5, 0.1)  
        y_tmp = np.arange(-5, 5, 0.1)  
        Z = np.empty((len(x_tmp), len(y_tmp)))
        fx.sort()
        
        for i, x_val in enumerate(x_tmp):
            for j, y_val in enumerate(y_tmp):
                Z[i, j] = f(np.array([x_val, y_val]))
        
        plt.contour(x_tmp, y_tmp, Z, levels=fx)
        
        for ind, val in enumerate(x):
            if ind != len(x) - 1:
                dx = 0.25 * (x[ind + 1][0] - val[0]) / (np.linalg.norm(x[ind + 1] - val))
                dy = 0.25 * (x[ind + 1][1] - val[1]) / (np.linalg.norm(x[ind + 1] - val))
                plt.arrow(val[0], val[1], dx, dy, head_width=0.2, width=0.05)
        
        plt.savefig(f"plots/{f.__name__}{np.array2string(initial_point)}_{condition}_cont.png")
        plt.close()

def bisection(x0:NDArray[np.float64],f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],d_k:NDArray[np.float64]):
    # ...
    c1 = 0.001
    c2 = 0.1
    alpha0 = 0
    t = 1
    beta0 = 1e6
    # eps = 1e-6
    max_iter = 1000
    x = x0
    for _ in range(max_iter):
        if(f(x+t*d_k)>f(x)+c1*t*np.dot(d_f(x),d_k)):
            beta0 = t 
            t = (alpha0+beta0)/2
        elif(np.dot(d_f(x+t*d_k),d_k)<c2*np.dot(d_f(x),d_k)):
            alpha0 = t
            t = (alpha0+beta0)/2
        else:
            break
    return t
def find_beta(approach,d_k,g_k,g_k1):
    # used to calculate beta values for different set of approaches
    epse = 1e-9
    if(approach == "Hestenes-Stiefel"):
        return np.dot(g_k1,(g_k1-g_k))/(np.dot(d_k,(g_k1-g_k))+epse)
    elif(approach == "Polak-Ribiere"):
        return np.dot(g_k1,(g_k1-g_k))/(np.dot(g_k,g_k)+epse)
    elif(approach == "Fletcher-Reeves"):
        return np.dot(g_k1,g_k1)/(np.dot(g_k,g_k)+epse)
def conjugate_descent(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    approach: Literal["Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves"],
) -> NDArray[np.float64]:
    # ...
    x_k = inital_point
    k = 0
    g0 = d_f(x_k)
    d0 = -g0
    g_k = g0
    d_k = d0
    eps = 1e-6
    n = x_k.shape[0] # dimension
    a = 0 
    max_iter = 1000
    fx_vals = []
    grad_fx_vals = []
    x = []
    while(np.linalg.norm(g_k)>eps and a<max_iter):
        # print(inital_point)
        fx_vals.append(f(x_k))
        grad_fx_vals.append(g_k)
        x.append(x_k)
        a +=1
        alpha_k = bisection(x_k,f,d_f,d_k)
        x_k =x_k + alpha_k*d_k
        g_k1 = d_f(x_k)
        if(k < n-1):
            # print(k)
            # different approaches were only differed in calculating beta values 
            # so defined a function called find_beta for different approaches
            beta_k = find_beta(approach,d_k,g_k,g_k1)
            d_k = -g_k1 + beta_k*d_k
            k = k+1
        else:
            d_k = -g_k1
            k = 0
        g_k =g_k1
    # print(len(fx_vals))
    fx_vals.append(f(x_k))
    grad_fx_vals.append(g_k)
    x.append(x_k)
    # plot_graphs(approach,fx_vals,grad_fx_vals,f,inital_point,x)
    return x_k

def sr1(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    x0 = initial_point
    eps = 1e-6
    k = 0 
    x_k = x0
    g_k = d_f(x_k)
    B_k = np.eye(len(x_k))
    max_iter = 1000
    fx_vals = []
    grad_fx_vals = []
    x = []
    epse = 1e-9
    while(np.linalg.norm(g_k) > eps and k < max_iter):
        fx_vals.append(f(x_k))
        grad_fx_vals.append(g_k)
        x.append(x_k)
        d_k = -B_k @ g_k
        alpha_k = bisection(x_k, f, d_f, d_k)
        x_k1 = x_k + alpha_k * d_k
        g_k1 = d_f(x_k1)
        gamma_k = g_k1 - g_k
        del_k = x_k1 - x_k
        mat1 = del_k - B_k @ gamma_k
        # if np.abs(gamma_k @ mat1) > 1e-8:
        B_k = B_k + np.outer(mat1, mat1) / (gamma_k @ mat1 + epse)
        g_k = g_k1
        x_k = x_k1
        k = k + 1
    fx_vals.append(f(x_k))
    grad_fx_vals.append(g_k)
    x.append(x_k)
    plot_graphs("SR1",fx_vals,grad_fx_vals,f,initial_point,x)
    return x_k

def dfp(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    # x0 = inital_point
    eps = 1e-6
    k = 0 
    x_k = inital_point
    g_k = d_f(x_k)
    B_k = np.eye(len(x_k))
    max_iter = 1000
    fx_vals = []
    grad_fx_vals = []
    x = []
    epse = 1e-9
    while(np.linalg.norm(g_k)>eps and k<max_iter):
        fx_vals.append(f(x_k))
        grad_fx_vals.append(g_k)
        x.append(x_k)
        d_k = -B_k@g_k
        alpha_k = bisection(x_k,f,d_f,d_k)
        x_k1 = x_k + alpha_k*d_k
        g_k1 = d_f(x_k1)
        gamma_k =g_k1 - g_k
        del_k = x_k1 -x_k
        
        B_k += np.outer(del_k, del_k) / (np.dot(del_k, gamma_k)+epse) - \
            np.outer(np.dot(B_k, gamma_k), np.dot(B_k, gamma_k)) / (np.dot(gamma_k, np.dot(B_k, gamma_k))+epse)
        g_k = g_k1
        x_k = x_k1
        k = k+1
    fx_vals.append(f(x_k))
    grad_fx_vals.append(g_k)
    x.append(x_k)
    # print(len(fx_vals))
    plot_graphs("DFP",fx_vals,grad_fx_vals,f,inital_point,x)
    return x_k

def bfgs(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    x0 = initial_point
    eps = 1e-6
    k = 0 
    x_k = x0
    g_k = d_f(x_k)
    B_k = np.eye(len(x_k))
    max_iter = 1000
    fx_vals = []
    grad_fx_vals = []
    x = []
    epse = 1e-9
    while(np.linalg.norm(g_k) > eps and k < max_iter):
        fx_vals.append(f(x_k))
        grad_fx_vals.append(g_k)
        x.append(x_k)
        d_k = -np.dot(B_k, g_k)
        alpha_k = bisection(x_k, f, d_f, d_k)
        x_k1 = x_k + alpha_k * d_k
        g_k1 = d_f(x_k1)
        gamma_k = g_k1 - g_k
        del_k = x_k1 - x_k
        delta_xk = del_k.reshape(-1,1)
        delta_gk = gamma_k.reshape(-1,1)
        term1 = (1 + (((delta_gk.T)@(B_k)@delta_gk)/(delta_gk.T@delta_xk)))*((delta_xk@delta_xk.T)/(delta_xk.T@delta_gk))
        term2 = ((B_k@delta_gk@delta_xk.T) + (B_k@delta_gk@delta_xk.T).T)/(delta_gk.T@delta_xk)
        B_k = (B_k + term1 - term2)
        g_k = g_k1
        x_k = x_k1
        k += 1
    fx_vals.append(f(x_k))
    grad_fx_vals.append(g_k)
    x.append(x_k)
    plot_graphs("BFGS",fx_vals,grad_fx_vals,f,initial_point,x)
    return x_k



