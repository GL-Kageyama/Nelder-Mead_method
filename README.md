# Implementation comparison of Nelder-Mead method
It is three patterns of solution method in Nelder-Mead method.

## Pattern 1 : Basic Python implementation
iters :  250  
Estimated minimum value :  0.0001741777615634549  
x:  [1.00021535 1.00009805 0.99896382 0.99809107 0.99638181]  

## Pattern 2 : Implementation with Scipy API
Solution :  [0.99910115 0.99820923 0.99646346 0.99297555 0.98600385]  
Optimal value :  6.617481708884532e-05  
iter :  141  

## Pattern 3 : Implementation of BFGS method by Scipy.optimize.minimize
Solusion :  [1.00000004 1.0000001  1.00000021 1.00000044 1.00000092]  
Optimal vallue :  4.0130879949972905e-13  
iter :  25  
