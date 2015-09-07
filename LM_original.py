# The Levenberg Marquardt algorithm
# Numpy and Numdifftools are used for matrix and derivative computation

import numpy as np
import numdifftools as nd

# compute the Jacobian matrix of function 'f' at point 'xk'
def Jacobian(f,xk):
        i=0
        n=len(xk)
        J=np.empty([0,n])
        a=np.empty([1,n])
        while i+1<= len(f):
            grd= nd.Jacobian(f[i])
            a[0:n-1] = grd(xk)
            row=np.array(a)
            i=i+1
            J=np.concatenate((J,row),axis=0)
        return J

# compute the object function value and the gradient at current point p
def function(p,J):
    r = np.array([10*(p[1]-p[0]**2),(1-p[0])])   
    fp = np.dot(np.transpose(r),r) # = 100*(p[1]-p[0]**2)**2 + (1-p[0])**2   namely r^T r
    grad = np.dot(J.T,r.T)
    return fp,r,grad

# LM algorithm
def lm(f,p0,tol=10**(-5),maxits=100):          #p0: initial point;  tol: grad norm accuracy;  maxits: maximum number of iterations
    nvars=np.shape(p0)[0]  # obtain the size of p
    nu=0.01          # the damping factor
    p = p0           # initialize the starting point
    J = Jacobian(f,p)
    fp,r,grad= function(p,J)     
    e = np.sum(np.dot(np.transpose(r),r))   # the sum of square error
    nits = 0     # the index of iterations
    while nits<maxits and np.linalg.norm(grad)>tol:      # control the loop: either the number of iterations is less than the maximum, or the optimization accuracy is achieved
        nits += 1
        J = Jacobian(f,p)
        fp,r,grad= function(p,J)     
        pnew = np.zeros(np.shape(p))  # define a new point with the same size of p
        nits2 = 0   # the index of iterations of the loop for controlling the trust region size or damping factor
        print "Iteration "+str(nits)+":"
        
        while (p!=pnew).all() and nits2<maxits:
            nits2 += 1
            H=np.dot(np.transpose(J),J) + nu*np.eye(nvars)  # obtain the Hessian approximation 
            dp,resid,rank,s = np.linalg.lstsq(H,-grad)   # compute the descent direction 'dp'
            pnew = p + dp  # obtain the new point
            Jnew = Jacobian(f,pnew) 
            fpnew,rnew,gradnew= function(pnew,Jnew) 
            enew = np.sum(np.dot(np.transpose(rnew),rnew)) 
            ared = np.dot(np.transpose(r),r)-np.dot(np.transpose(rnew),rnew) # actual reduction
            pred = np.dot(np.transpose(grad),p-pnew)  # predict reduction
            rho = ared/pred  # compute the ratio
            print "    lambda:"+str(nu)
            print "        ratio: "+str(rho)
            if rho>0:
                p = pnew   # take the new point
                e = enew   
                if rho>0.25:  
                    nu=nu/10  # reduce the damping factor is to increase the trust region
                    print "        ratio>0.25: new step accepted, reduce lambda"
                else:
                    print "        0<ratio<0.25: , new step accepted, lambda remain unchanged"
            else:   
                nu=nu*10 # increase the factor is to reduce the trust region
                print "        ratio<0: new step rejected, increase lambda"  
        print "   ","Fun Val: ".ljust(20), "Solution: ".ljust(40), "LSQ Err: ".ljust(20), "Accuracy: ".ljust(20), "lambda: ".ljust(20)
        print "   ",str(fp).ljust(20), str(p).ljust(40), str(e).ljust(20), str(np.linalg.norm(grad)).ljust(20), str(nu).ljust(20)
        print

f = [lambda x: 10*(x[1]-x[0]**2), lambda x: (1-x[0])] 
p0 = np.array([-1.92,2])
lm(f,p0)