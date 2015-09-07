import numpy as np
from scipy.special import expit
from scipy import signal
import pylab as pl
from mpl_toolkits.mplot3d import axes3d

# the derivative of sigmod(x)
def dsigmoid(x):
    return expit(x)*(1-expit(x))

# feedforward calculation w.r.t. a single input pattern
def output(p,W1,W2,b1,b2,S0):
    a0 = p.reshape(S0,1)
    n1 = np.dot(W1,a0)+b1
    a1 = expit(n1)
    n2 = np.dot(W2,a1)+b2
    a2 = n2
    return n1,n2,a1,a2

# feedforward calculation w.r.t. all input patterns
def simulation(x,W1,W2,b1,b2,S0):
    Q=np.shape(x)[1]
    i=0
    simu=np.zeros(Q,)
    while i<Q:
         n1,n2,a1,out=output(x[:,i],W1,W2,b1,b2,S0)
         np.put(simu,i,out)
         i=i+1
    return simu

# backparopagatoin w.r.t. a single input pattern
def backpropagation(p,e,n1,n2,a1,a2,W1,W2,SM,S0):
    Ff = np.ones([1,1])
    delta2=-Ff
    db2=delta2
    f = dsigmoid(n1)  
    F = np.diag(np.squeeze(f))
    delta1=np.dot(F,W2.T)*delta2
    db1=delta1
    dW1=np.dot(delta1,p.reshape(1,S0))
    dW2=np.reshape(np.dot(a1,delta2),np.shape(W2))
    dW1=dW1.flatten()
    dW2=dW2.flatten()
    db1=db1.flatten()
    db2=db2.flatten()
    Jr=np.append(dW1,dW2,axis=1)
    Jr=np.append(Jr,db1,axis=1)
    Jr=np.append(Jr,db2,axis=1)
    Jr =Jr.reshape(1,SM)
    return Jr
    
#  reconstruct the descent direction of weights and bias from the Jacobian  
def reconsctruct(Jr,W1,W2,b1,b2):
    rW1=np.shape(W1)[0]
    cW1=np.shape(W1)[1]
    rW2=np.shape(W2)[0]
    cW2=np.shape(W2)[1]
    rb1=np.shape(b1)[0]
    cb1=np.shape(b1)[1]
    rb2=np.shape(b2)[0]
    cb2=np.shape(b2)[1]
    start=0
    end=rW1*cW1
    dW1=np.reshape(Jr[start:end],np.shape(W1))
    start=end
    end=end+rW2*cW2
    dW2=np.reshape(Jr[start:end],np.shape(W2))
    start=end
    end=end+rb1*cb1
    db1=np.reshape(Jr[start:end],np.shape(b1))
    start=end
    end=end+rb2*cb2-1
    db2=np.reshape(Jr[end],np.shape(b2))
    return dW1,dW2,db1,db2

# Levenberg-Marquardt algorithm    
def LM(x,y,S0,S1,S2,maxits=100,tol=0.02,nu=0.01):
    W1 = np.random.normal(0.0, 1.0, [S1,S0])
    W2 = np.random.normal(0.0, 1.0, [S2,S1])
    b1= np.random.normal(0.0,1.0,[S1,1])
    b2= np.random.normal(0.0,1.0,[S2,1])
    Q = np.shape(x)[1]
    SM = np.shape(W1)[0]*np.shape(W1)[1]+np.shape(W2)[0]*np.shape(W2)[1]+np.shape(b1)[0]*np.shape(b1)[1]+np.shape(b2)[0]*np.shape(b2)[1]
    ys=simulation(x,W1,W2,b1,b2,S0)
    e=y-ys
    print "error:"+str(np.linalg.norm(e))
    print
    epoch=0
    while np.linalg.norm(e)>tol and epoch<maxits:
        i=0
        J=np.zeros([0,SM])
        print "epoch"+str(epoch+1)+":"
        while i<Q:
            n1,n2,a1,a2=output(x[:,i],W1,W2,b1,b2,S0)
            Jr = backpropagation(x[:,i],e[i],n1,n2,a1,a2,W1,W2,SM,S0)  # one row of Jacobian matrix
            J=np.concatenate((J,Jr),axis=0)
            i=i+1
        nits2=0
        enew=np.zeros(np.shape(e))
        while (e!=enew).all() and nits2<maxits:
            nits2+=1
            H=np.dot(np.transpose(J),J) + nu*np.eye(SM)
            dx,resid,rank,s = np.linalg.lstsq(H,-np.dot(J.T,e))
            dW1,dW2,db1,db2 = reconsctruct(dx,W1,W2,b1,b2)
            W1n=W1+dW1
            W2n=W2+dW2
            b1n=b1+db1
            b2n=b2+db2
            enew=y-simulation(x,W1n,W2n,b1n,b2n,S0)
            if np.linalg.norm(enew)<np.linalg.norm(e):
                W1=W1n
                W2=W2n
                b1=b1n
                b2=b2n
                e=enew
                nu=nu/10
            else:
                nu=nu*10
        print "  error: "+str(np.linalg.norm(e))
        print "  lambda: "+str(nu)
        epoch+=1
    return W1,W2,b1,b2

# generate the data for the four examples    
def data_generation(S0,example):
    if example==1:
        x = np.arange(-1,1,0.05)
        y = 0.5+0.25*np.sin(3*np.pi*x)
        x = np.atleast_2d(x)
    if example==2:
        x = np.arange(-1,1,0.05)
        y=signal.square(2*np.pi*x,0.5)
        x = np.atleast_2d(x)  
    if example==3:
        axisx=np.arange(-2,2,0.1)
        axisy=np.arange(-2,2,0.1)
        x=np.zeros([S0,1])
        i=0
        while i<len(axisx):
            j=0
            while j<len(axisy):
                x=np.concatenate((x,np.reshape([axisx[i],axisy[j]],(S0,1))),axis=1)
                j+=1
            i+=1
        X,Y=np.meshgrid(axisx,axisy)
        z = np.sinc(X)*np.sinc(Y) 
        y = z.flatten()
        x = np.delete(x,0,1)
    if example==4:
        Q=400
        #x=np.random.normal(0, 0.5, [S0,Q])
        x=np.random.uniform(-1,1,(S0,Q))
        y=np.zeros([Q,])
        i=0
        while i<Q:
            y[i]=np.sin(2*np.pi*x[0,i])*x[1,i]**2*x[2,i]**3*x[3,i]**4*np.exp(-x[0,i]-x[1,i]-x[2,i]-x[3,i])
            i+=1
    return x,y

# plot the data for the four examples
def data_plot(x,y,yt,example):
    if example==1:
        pl.plot(np.squeeze(x),y,'r')
        pl.plot(np.squeeze(x),yt,'g--')
        pl.legend(['Actual Data','Fitted Data'])
        pl.show()
    if example==2:
        pl.plot(np.squeeze(x),y,'r')
        pl.plot(np.squeeze(x),yt,'g--')
        pl.legend(['Actual Data','Fitted Data'])
        pl.show()
    if example==3:
        total=np.shape(x)[1]
        length=np.sqrt(total)
        input_variable=np.zeros([length,])
        i=0
        while i<length:
            input_variable[i]=x[0,i*length]
            i+=1
        y=y.reshape(length,length)
        yt=yt.reshape(length,length)
            
        X,Y=np.meshgrid(input_variable,input_variable)
        fig = pl.figure(0)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, y)
        pl.show()
        pl.title('Actual Function')
        
        fig = pl.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, yt)
        pl.show()
        pl.title('Fitted Function')
    if example==4:
        print
        print "Sorry, cannot plot 4-D function!"
    return 


example=4 #1 #2 #3 #4
S0 = 4   # the number of neurons in Layer 0: the input layer  
S1 = 50  # the number of neurons in Layer 1: the hidden layer
S2 = 1   # the number of neurons in Layer 2: the output layer

x,y=data_generation(S0,example)
W1,W2,b1,b2=LM(x,y,S0,S1,S2)
yt=simulation(x,W1,W2,b1,b2,S0)
data_plot(x,y,yt,example)


