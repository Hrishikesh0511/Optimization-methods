from typing import Callable, Literal
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
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
def do_backtracking(condition,x0:npt.NDArray[np.float64],f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]: 
    fx: list = []
    grad_fx: list = []
    x=[]
    # initialization
    alpha0 = 10.0
    rhow = 0.75
    c = 0.001
    k = 0
    eps = 1e-6
    grad_f:npt.NDArray[np.float64] = d_f(x0)
    x_k:npt.NDArray[np.float64] = x0
    while(k <= 10000 and np.linalg.norm(grad_f) > eps):
        fx.append(f(x_k))
        grad_fx.append(grad_f)
        x.append(x_k)
        alpha = alpha0
        d_k:npt.NDArray[np.float64] = - grad_f
        while(f(np.add(x_k,alpha*d_k)) > f(x_k) + c*alpha*(grad_f.T@d_k)):
            alpha *= rhow
        x_k = x_k + alpha*d_k
        grad_f = d_f(x_k)
        k += 1
    plot_graphs(condition,fx,grad_fx,f,x0,x)
    return x_k
def do_bisection(condition,x0:npt.NDArray[np.float64],f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    # initialization
    c1 = 0.001
    c2 = 0.1 
    alpha0 =0
    t = 1
    beta0 = 1e6
    k = 0
    eps = 1e-6
    fx=[]   # list to store all function values
    grad_fx=[] # list to store all gradients 
    x=[]    # list to store all x_k values
    x_k:npt.NDArray[np.float64] = x0
    grad_f:npt.NDArray[np.float64] = d_f(x0)
    while(k <= 1e4 and np.linalg.norm(grad_f) > eps):
        fx.append(f(x_k))
        grad_fx.append(grad_f)
        x.append(x_k)
        d_k:npt.NDArray[np.float64] = -grad_f
        alpha = alpha0
        beta = beta0
        while(True):
            if (f(np.add(x_k,t*d_k)) > f(x_k)+c1*t*(grad_f.T@d_k)):
                beta = t 
                t = 0.5*(alpha + beta)
            elif (d_f(np.add(x_k,t*d_k)).T@d_k < c2*(grad_f.T@d_k)):
                alpha = t
                t = 0.5*(alpha + beta)
            else:
                break
        x_k = x_k + t*d_k
        grad_f = d_f(x_k)
        k += 1
    plot_graphs(condition,fx,grad_fx,f,x0,x)
    return x_k
def do_pureNewton(condition,x0:npt.NDArray[np.float64],f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    x_k:npt.NDArray[np.float64] = x0
    k = 0
    eps = 1e-6
    grad_f:npt.NDArray[np.float64] = d_f(x0)
    fx=[]
    grad_fx=[]
    x=[]
    while(k <= 1e4 and np.linalg.norm(grad_f) > eps):
        fx.append(f(x_k))
        grad_fx.append(grad_f)
        x.append(x_k)
        d_k:npt.NDArray[np.float64] = np.linalg.solve(d2_f(x_k),-grad_f)
        x_k = x_k + d_k
        grad_f = d_f(x_k)
        k+=1
    plot_graphs(condition,fx,grad_fx,f,x0,x)
    return x_k
def do_dampedNewton(condition,x0:npt.NDArray[np.float64],f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    # initialization
    alpha = 0.001
    beta = 0.75
    x_k:npt.NDArray[np.float64] = x0
    k = 0
    eps = 1e-6
    grad_f:npt.NDArray[np.float64] = d_f(x0)
    fx=[]
    grad_fx=[]
    x=[]
    while(k <= 1e4 and np.linalg.norm(grad_f) > eps):
        fx.append(f(x_k))
        grad_fx.append(grad_f)
        x.append(x_k)
        d_k:npt.NDArray[np.float64] = np.linalg.solve(d2_f(x_k),-grad_f)
        t_k = 1
        while(f(x_k)-f(np.add(x_k,t_k*d_k)) < -alpha*t_k*(grad_f.T@d_k)):
            t_k *= beta
        x_k = x_k + t_k*d_k
        grad_f = d_f(x_k)
        k += 1
    plot_graphs(condition,fx,grad_fx,f,x0,x)
    return x_k
def do_levenbergMarquardt(condition,x0:npt.NDArray[np.float64],f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    # initialization
    k = 0
    eps = 1e-6
    x_k:npt.NDArray[np.float64] = x0
    grad_f:npt.NDArray[np.float64] = d_f(x0)
    hess_f:npt.NDArray[np.float64] = d2_f(x0)
    fx=[]
    grad_fx=[]
    x=[]
    while(k <= 1e4 and np.linalg.norm(grad_f) > eps):
        fx.append(f(x_k))
        grad_fx.append(grad_f)
        x.append(x_k)
        lambda_min = np.min(np.linalg.eigvals(hess_f))
        if lambda_min <= 0:
            myu_k = -lambda_min + 0.1
            d_k = -np.linalg.inv(hess_f + myu_k*np.eye(hess_f.shape[0],hess_f.shape[1]))@grad_f
        else:
            d_k = -np.linalg.inv(hess_f)@grad_f
        x_k = x_k + d_k
        grad_f = d_f(x_k)
        hess_f = d2_f(x_k)
        k+=1
    plot_graphs(condition,fx,grad_fx,f,x0,x)
    return x_k
def do_combinedNewton(condition,x0:npt.NDArray[np.float64],f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    # initialization
    k = 0
    eps = 1e-6
    x_k:npt.NDArray[np.float64] = x0
    grad_f:npt.NDArray[np.float64] = d_f(x0)
    hess_f:npt.NDArray[np.float64] = d2_f(x0)
    fx=[]
    grad_fx=[]
    x=[]
    while(k <= 1e4 and np.linalg.norm(grad_f) > eps):
        fx.append(f(x_k))
        grad_fx.append(grad_f)
        x.append(x_k)
        lambda_min = np.min(np.linalg.eigvals(hess_f))
        if lambda_min <= 0:
            myu_k = -lambda_min + 0.1
            d_k = -np.linalg.inv(hess_f + myu_k*np.eye(hess_f.shape[0],hess_f.shape[1]))@grad_f
        else:
            d_k = -np.linalg.inv(hess_f)@grad_f
        # backtracking method to find alpha_k
        # using backtracking initial configs
        alpha_k = 10
        rhow = 0.75
        c = 0.001
        while(f(np.add(x_k,alpha_k*d_k)) > f(x_k) + c*alpha_k*(grad_f.T@d_k)):
            alpha_k *= rhow
        x_k = x_k + alpha_k*d_k
        grad_f = d_f(x_k)
        hess_f = d2_f(x_k)
        k+=1
    plot_graphs(condition,fx,grad_fx,f,x0,x)
    return x_k
# Do not rename or delete this function
def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking", "Bisection"],
) -> npt.NDArray[np.float64]:
    # Complete this function
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_vals.png" for plotting f(x) vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_grad.png" for plotting |f'(x)| vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_cont.png" for plotting the contour plot
    # f-> actual function, d_f-> derivative 
    if condition == "Backtracking":
        return do_backtracking(condition,inital_point,f,d_f)
    elif condition == "Bisection":
        return do_bisection(condition,inital_point,f,d_f)


# Do not rename or delete this function
def newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Pure", "Damped", "Levenberg-Marquardt", "Combined"],
) -> npt.NDArray[np.float64]:
    # Complete this function
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_vals.png" for plotting f(x) vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_grad.png" for plotting |f'(x)| vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_cont.png" for plotting the contour plot
    if condition == "Pure":
        return do_pureNewton(condition,inital_point,f,d_f,d2_f)
    elif condition == "Damped":
        return do_dampedNewton(condition,inital_point,f,d_f,d2_f)
    elif condition == "Levenberg-Marquardt":
        return do_levenbergMarquardt(condition,inital_point,f,d_f,d2_f)
    else:
        return do_combinedNewton(condition,inital_point,f,d_f,d2_f)