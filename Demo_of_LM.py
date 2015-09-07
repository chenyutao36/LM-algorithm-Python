
# Code from Chapter 11 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

# The Levenberg Marquardt algorithm solving a least-squares problem

#import pylab as pl
import numpy as np
import numdifftools as nd

def Jacobian(x,y,xk):
    m=int(x.shape[0])
    f=np.empty([1,],dtype=object)
    i=0
    while i < m:    
        g=[lambda z: y[0,i]-z[0]*np.cos(z[1]*x[i])-z[1]*np.sin([z[0]*x[i]])]
        i=i+1
        f=np.concatenate((f,g),axis=0)
    f=f[1:]  
    n=len(xk)  
    i=0
    J=np.empty([0,n])
    row=np.empty([1,n])
    while i+1<= m:
        grd= nd.Jacobian(f[i])
        row[0:n-1] = grd(xk)
        i=i+1
        J=np.concatenate((J,row),axis=0)
    return J


def function(p,J,x,ydata):
    fp = p[0]*np.cos(p[1]*x)+ p[1]*np.sin([p[0]*x])
    r = ydata - fp
    grad = np.dot(J.T,r.T)
    return fp,r,grad

def rootfinding(J,p,Delta,grad):
    nvars=np.shape(p)[0] 
    nits2=0
    nu=0.01
    dnu = 1
    maxits = 100
    while np.absolute(dnu)>10**(-2) and nits2<maxits/10:
        nits2 += 1
        L = np.linalg.cholesky(np.dot(np.transpose(J),J) + nu*np.eye(nvars))
        R = [L[0,0],L[0,1]],[L[1,0],L[1,1]]
        R = np.array(R)
        pl,residp,rankp,sp = np.linalg.lstsq(np.dot(np.transpose(R),R),-grad)
        ql,residq,rankq,sq = np.linalg.lstsq(np.transpose(R),pl)
        dnu = (np.linalg.norm(pl)/np.linalg.norm(ql))**2*(np.linalg.norm(pl)-Delta)/Delta
        nu = nu + dnu
    return nu,pl

def lm(p0,x,y,tol=10**(-5),maxits=100):

    nvars=np.shape(p0)[0]
    nu=0.01
    Delta=1
    p = p0
    J = Jacobian(x,y,p)
    fp,r,grad= function(p,J,x,y)
    e = np.sum(np.dot(np.transpose(r),r))
    nits = 0
    while nits<maxits and np.linalg.norm(grad)>tol:
        nits += 1
        J = Jacobian(x,y,p)
        fp,r,grad = function(p,J,x,y)
        pnew = np.zeros(np.shape(p))
        nits2 = 0
        print "Iteration "+str(nits)+":"
        while (p!=pnew).all() and nits2<maxits:
            nits2 += 1
            H=np.dot(np.transpose(J),J) + nu*np.eye(nvars)
            dp,resid,rank,s = np.linalg.lstsq(H,grad)
            #dp = -dot(linalg.inv(H),dot(transpose(J),transpose(d)))
            pnew = p - dp[:,0]
            
            # Decide whether the trust region is good
            Jnew = Jacobian(x,y,pnew)
            fpnew,rnew,gradnew = function(pnew,Jnew,x,y)
            enew = np.sum(np.dot(np.transpose(rnew),rnew))
            
            ardct = np.linalg.norm(np.dot(np.transpose(r),r)-np.dot(np.transpose(rnew),rnew))
            prdct = np.linalg.norm(np.dot(np.transpose(grad),pnew-p))
            rho = ardct / prdct
            #print "   ardct: "+str(ardct)
            #print "   prdct: "+str(prdct)
            print "   Ratio: "+str(rho)
                           
            if rho>0:
                # Keep new estimate
                p = pnew
                e = enew
                if rho>0.25:
                    # Make trust region larger (reduce nu)
                    nu=nu/10
            else: 
                # Make trust region smaller (increase nu)
                #nu=nu*10
                nu,p=rootfinding(J,p,Delta,grad)
                pnew=p
        print "   ","Solution: ".ljust(40), "LSQ Err: ".ljust(20), "Gradient: ".ljust(20), "Damping Factor: ".ljust(20)
        print "   ",str(p).ljust(40), str(e).ljust(20), str(np.linalg.norm(grad)).ljust(20), str(nu).ljust(20)
        print
    return p

    
p0 = np.array([100.5,102.5]) #[ 100.0001126   101.99969709] 1078.36915936 8.87386341319e-06 1e-10 (8 itns)
#p0 = np.array([101,101]) #[ 100.88860713  101.12607589] 631.488571159 9.36938417155e-06 1e-67

p = np.array([100,102])

x = np.arange(0,2*np.pi,0.1)
y = p[0]*np.cos(p[1]*x)+ p[1]*np.sin([p[0]*x]) + np.random.rand(len(x))

p = lm(p0,x,y)
"""
y1 = p[0]*np.cos(p[1]*x)+ p[1]*np.sin([p[0]*x]) #+ np.random.rand(len(x))


pl.plot(x,np.squeeze(y),'-')
pl.plot(x,np.squeeze(y1),'r--')
pl.legend(['Actual Data','Fitted Data'])
pl.show()
"""