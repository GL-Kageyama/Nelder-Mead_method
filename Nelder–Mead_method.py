#====================================================================
#------------------       Nelder-Mead method     --------------------
#====================================================================
#It is three patterns of solution method in Nelder-Mead method.
#Pattern 1 : Basic Python implementation
#====================================================================

import copy
import numpy as np

def get_x0(init_x, step=0.1):
    dim = len(init_x)
    x0 = [init_x]
    for i in range(dim):
        x = copy.copy(init_x)
        x[i] = x[i] + step
        x0.append(x)
    return np.array(x0)

def get_centroid(set_x_except_max_point):
    return np.sum([x for x, y in set_x_except_max_point], axis=0)/len(set_x_except_max_point)

def get_centroid(set_x_y_except_max_point):
    _set_x_y_except_max_point = np.array(set_x_y_except_max_point)
    
    return np.sum(_set_x_y_except_max_point[:,0]/len(set_x_y_except_max_point))
    
def get_reflection_point(worst_point, centroid, alpha=1.0):
    return centroid + alpha*(centroid - worst_point)

def get_expansion_point(worst_point, centroid, beta=2.0):
    return centroid + beta*(centroid - worst_point)

def get_outside_contraction_point(worst_ponit, centroid, gamma=0.5):
    return centroid + gamma*(centroid - worst_ponit)

def get_inside_contraction_ponit(worst_point, centroid, gamma=0.5):
    return centroid - gamma*(centroid - worst_point)

def get_shrinkage_point(set_x_y, best_point, delta=0.5):
    return [best_point + delta*(x - best_point) for x, y in set_x_y]

def core_algorithm_in_nelder_mead(func, set_x_y):
    best_point = set_x_y[0][0]
    best_score = set_x_y[0][1]
    
    worst_point = set_x_y[-1][0]
    worst_score = set_x_y[-1][1]
    
    second_worst_point = set_x_y[-2][0]
    second_worst_score = set_x_y[-2][1]
    
    centroid = get_centroid(set_x_y[:-1])
    
    reflection_point = get_reflection_point(worst_point, centroid)
    reflection_score = func(reflection_point)
    
    if best_score <= reflection_score < second_worst_score:
        del set_x_y[-1]
        set_x_y.append([reflection_point, reflection_score])
        return set_x_y
    
    elif reflection_score < best_score:
        expansion_point = get_expansion_point(worst_point, centroid)
        expansion_score = func(expansion_point)
        
        if expansion_score < reflection_score:
            del set_x_y[-1]
            set_x_y.append([expansion_point, expansion_score])
            
            return set_x_y
        else:
            del set_x_y[-1]
            set_x_y.append([reflection_point, reflection_score])
            return set_x_y
        
    elif second_worst_score <= reflection_score:
        outside_contraction_point = get_outside_contraction_point(worst_point, centroid)
        outside_contraction_score = func(outside_contraction_point)
        
        if outside_contraction_score < worst_score:
            del set_x_y[-1]
            set_x_y.append([outside_contraction_point, outside_contraction_score])
            return set_x_y
        
    shrinkage_point_list = get_shrinkage_point(set_x_y, best_point)
    shrinkage_score_list = [func(reduction_point) for reduction_point in shrinkage_point_list]
    
    reduction_value = zip(shrinkage_point_list, shrinkage_score_list)
    
    return reduction_value

def get_solution_by_nelder_mead(func, init_x, no_improve_thr=10e-8, no_improv_break=10, max_iter=0):
    set_x = get_x0(init_x)
    
    set_y = [func(x) for x in set_x]
    
    set_x_y = zip(set_x, set_y)
    set_x_y = sorted(set_x_y, key=lambda t: t[1])
    
    prev_best_score = set_x_y[0][1]
    
    is_not_sarturate = True
    no_improv = 0
    iters = 0
    while is_not_sarturate:
        best_value = set_x_y[0]
        best_score = set_x_y[0][1]
        
        if max_iter and iters >= max_iter:
            print("iters : ", iters)
            return best_value
        
        if best_score < prev_best_score - no_improve_thr:
            no_improv = 0
            prev_best_score = best_score
        else:
            no_improv += 1
            
        if no_improv >= no_improv_break:
            print("=====Basic Python implementation=====")
            print("iters : ", iters)
            return best_value
        
        prev_best_score = set_x_y[0][1]
        
        set_x_y = core_algorithm_in_nelder_mead(func, set_x_y)
        
        set_x_y = sorted(set_x_y, key=lambda t: t[1])
        
        iters += 1

def rosen(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

res = get_solution_by_nelder_mead(rosen, x0)

print("Estimated minimum value : ", res[1])
print("x: ", res[0])
print("")

#====================================================================
#Pattern 2 : Implementation with Scipy API
#====================================================================

from scipy.optimize import minimize

res_by_scipy = minimize(rosen, x0, method="Nelder-Mead")

type(res_by_scipy)

print("=====Implementation with Scipy API=====")
print("Solution : ", res_by_scipy.x)
print("Optimal value : ", res_by_scipy.fun)
print("iter : ", res_by_scipy.nit)
print("")

#====================================================================
#Pattern 2 : Implementation of BFGS method by Scipy.optimize.minimize
#====================================================================

def rosen_deriv(x):
    x = np.asarray(x)
    xi = x[1:-1]
    xi_m1 = x[:-2]
    xi_p1 = x[2:]
    
    deriv = np.zeros_like(x)
    
    deriv[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    
    deriv[1:-1] = (200 * (xi - xi_m1**2) - 400 * (xi_p1 - xi**2) * xi - 2 * (1 - xi))
    
    deriv[-1] = 200 * (x[-1] - x[-2]**2)
    
    return deriv

res_BFGS = minimize(rosen, x0, method="BFGS", jac=rosen_deriv)

print("=====Implementation of BFGS method by Scipy.optimize.minimize=====")
print("Solusion : ", res_BFGS.x)
print("Optimal vallue : ", res_BFGS.fun)
print("iter : ", res_BFGS.nit)
print("")

