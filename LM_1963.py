# -*- coding: utf-8 -*-
# The LM algorithm using the strategy proposed by D.W.Marquardt(1963)
# Two packages-Numpy and Numdifftools are used to help with matrix and gradient computations
# This program is tested successfully on Canopy with Python 2.7

import numpy as np
import numdifftools as nd

def Jacobian(f,xk):
        i=0
        n=len(xk)   # the number of function variables
        J=np.empty([0,n])
        a=np.empty([1,n])
        while i+1<= len(f):
            grd= nd.Jacobian(f[i])
            a[0:n-1] = grd(xk)
            row=np.array(a)
            i=i+1
            J=np.concatenate((J,row),axis=0)
        return J  # obtain the Jacobian matrix

def function(p,J):
    r = np.array([10*(p[1]-p[0]**2),(1-p[0])])   # here r is a row vector
    fp = np.dot(np.transpose(r),r) #= 100*(p[1]-p[0]**2)**2 + (1-p[0])**2   namely r^T*r
    grad = np.dot(J.T,r.T)  # gradient
    return fp,r,grad

def lm(f,p0,tol=10**(-5),maxits=100,v=2):          #p0: initial point;  tol: gradient norm accuracy;  maxits: maximum number of iterations
    nvars=np.shape(p0)[0]  # obtain the size of p
    nu=0.01          # the damping factor
    p = p0           # initialize the starting point
    J = Jacobian(f,p) # calculate the Jacobian of the function at point p
    fp,r,grad= function(p,J)     # obtain the function, gradient and Jacobian at the starting point 
    e = np.sum(np.dot(np.transpose(r),r))  # obtain the sum of square errors
    nits = 0     # the index of iterations
    
    while nits<maxits and np.linalg.norm(grad)>tol:      # control the loop: either the number of iterations is less than the maximum, or the optimization accuracy is achieved
        nits += 1
        J = Jacobian(f,p)
        fp,r,grad= function(p,J)     
        pnew = np.zeros(np.shape(p))  # create a variable to store the new point
        nits2 = 0
        print "Iteration "+str(nits)+":"
        while (p!=pnew).all() and nits2<maxits/10 and nu!=0:
            nits2 += 1
            
            # Calculate the info of the new point
            H =np.dot(np.transpose(J),J) + nu*np.eye(nvars)  # calculate the LHS of the normal equation
            dp,resid,rank,s = np.linalg.lstsq(H,-grad)  # solve the linear normal equation
            pnew = p + dp   
            Jnew = Jacobian(f,pnew) 
            fpnew,rnew,gradnew= function(pnew,Jnew)
            enew = np.sum(np.dot(np.transpose(rnew),rnew))
            
            # Calculate the info of the last point with a smaller damping factor
            Hv =  np.dot(np.transpose(J),J) + nu/v*np.eye(nvars)  
            dpv,residv,ranvk,sv = np.linalg.lstsq(Hv,-grad)
            pv = p + dpv
            Jv = Jacobian(f,pv) 
            fpv,rv,gradv= function(pv,Jv)
            ev = np.sum(np.dot(np.transpose(rv),rv))
            
            # Factor selection strategy
            print "    lambda: "+str(nu)
            # Try a smaller factor until the error cannot be reduced
            if enew <= e:
                if ev <= e and ev < enew:
                    nu = nu / v
                else:
                    p = pnew
                    e = enew
            # The error is increased. Increase the factor and try again
            else:
                nu = nu * v   
            
        print "   ","Fun Val: ".ljust(20), "Solution: ".ljust(40), "LSQ Err: ".ljust(20), "Accuracy: ".ljust(20), "Damping Factor: ".ljust(20)
        print "   ",str(fp).ljust(20), str(p).ljust(40), str(e).ljust(20), str(np.linalg.norm(grad)).ljust(20), str(nu).ljust(20)
        print

f = [lambda x: 10*(x[1]-x[0]**2), lambda x: (1-x[0])]    # the function
p0 = np.array([300,2])    # the initial point
lm(f,p0)