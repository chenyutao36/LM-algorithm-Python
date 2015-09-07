# Back Propagation
# Steepest Descent

import numpy as np
from scipy.special import expit
from scipy import signal
import pylab as pl

def dsigmoid(x):
    return expit(x)*(1-expit(x))


def output(p,W1,W2,b1,b2):
    M = 2
    a0 = p
    k = 0
    while k< M-1:
        n1 = W1*a0+b1
        a1 = expit(n1)
        k = k+1
    n2 = np.dot(W2,a1)[0]+b2
    a2 = n2
    return n1,n2,a1,a2

def simulation(x,W1,W2,b1,b2):
    Q=len(x)
    i=0
    simu=np.zeros([Q,])
    while i<Q:
         n1,n2,a1,out=output(x[i],W1,W2,b1,b2)
         np.put(simu,i,out)
         i=i+1
    return simu

def backpropagation(p,e,n1,n2,a1,a2,W1,W2,alpha):

    Ff = 1
    delta2=-Ff*e
    db2=-alpha*delta2
    f = dsigmoid(n1)  
    F = np.diag(f)
    delta1=np.squeeze(np.dot(F,W2.T)*delta2)
    db1=-alpha*delta1
    dW1=-alpha*delta1*p
    dW2= np.atleast_2d(-alpha*delta2*a1) 
    return dW1,dW2,db1,db2
    
S1 = 15
S2 = 1
x = np.arange(-1,1,0.05)
x# = np.arange(0,6,0.05)
Q = len(x)
R = 1
y = 0.5+0.25*np.sin(3*np.pi*x)
#y=signal.square(2*np.pi*x,0.5)

W1 = np.random.normal(0.0, 1.0, [S1,])
W2 = np.random.normal(0.0, 1.0, [S2,S1])
b1= np.zeros([S1,])
b2= 0

dW1=np.zeros([S1,])
dW2=np.zeros([S2,S1])
db1=np.zeros([S1,])
db2=0

epoch=0
maxits=1e+3
ys=simulation(x,W1,W2,b1,b2)
e=y-ys

alpha=0.01
eta=0.1
print "error:"+str(np.linalg.norm(e))
print 
while np.linalg.norm(e)>0.02 and epoch<maxits:
    i=0
    print "epoch"+str(epoch+1)+":"
    while i<Q:
        n1,n2,a1,a2=output(x[i],W1,W2,b1,b2)
        error=y[i]-a2
        ddW1,ddW2,ddb1,ddb2=backpropagation(x[i],error,n1,n2,a1,a2,W1,W2,alpha)
        dW1=(1-eta)*dW1+eta*ddW1
        dW2=(1-eta)*dW2+eta*ddW2
        db1=(1-eta)*db1+eta*ddb1
        db2=(1-eta)*db2+eta*ddb2
        i=i+1
    W1=W1+dW1
    W2=W2+dW2
    b1=b1+db1
    b2=b2+db2
    ys=simulation(x,W1,W2,b1,b2)
    enew=y-ys
    #if np.linalg.norm(enew)>np.linalg.norm(e):
    #   alpha=alpha/10
    e=enew
    print "  error:"+str(np.linalg.norm(e))
    epoch=epoch+1

yt=simulation(x,W1,W2,b1,b2)
pl.plot(x,y,'b-')
#pl.axis([-1, 1, 0, 1])
pl.plot(x,yt,'g--')
pl.legend(['Actual Data','Fitted Data'])
pl.show()
