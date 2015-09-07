
# Code from Chapter 9 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Gradient Descent using steepest descent

import numpy as np
import numdifftools as nd
"""
def Jacobian(x):
    #return array([.4*x[0],2*x[1]])
    return np.array([x[0], 0.4*x[1], 1.2*x[2]])
"""
def Jacobian(f,xk,n):
    i=0
    #J=np.empty([0,n])
    x0=xk.tolist()
    while i+1<= 1:
        grd= nd.Jacobian(f)
        a, b = grd(x0)
        row=np.array([[a,b]])
        #J=np.concatenate((J,row),axis=0)
        #J=row
        i=i+1 
    return row

def steepest(r,x0):

    i = 0 
    iMax = 30
    x = x0
    Delta = 1
    alpha = 1
    
    while i<iMax and Delta>10**(-10):
        #p = -Jacobian(x)
        p=-Jacobian(r,x,2)
        xOld = x
        x = x + alpha*p
        Delta = np.sum((x-xOld)**2)
        print "Iteration"+str(i+1)
        print "    x="+str(x)+"  "+"Delta="+str(Delta)
        
        if Delta <=10**(-10):
            print "Performance goal achieved"
        i += 1
        
        if i == iMax:
            print "Maximum iterations achieved"
            
            
def r(x):
    return 0.5*x[0]**2+0.2*x[1]**2


x0 = np.array([-2,2])

steepest(r,x0)