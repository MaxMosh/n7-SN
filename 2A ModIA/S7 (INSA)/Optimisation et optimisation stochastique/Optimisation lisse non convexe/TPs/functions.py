import numpy as np

class square() :
    def __init__(self) :
        print("Fonction (x,y) --> x^2/2+7/2*y^2")
        self.zeros()
    def zeros(self) :
        self.nb_eval=0 # number of evaluations of the function self.value()
        self.nb_grad=0 # number of evaluations of the function self.grad()
        self.nb_hess=0 # number of evaluations of the function self.Hess()
    def value(self,x) :
        # returns the value of the function at point x
        self.nb_eval+=1
        return 0.5*x[0]**2+7/2.*x[1]**2
    def grad(self,x) :
        # returns the gradient of the function at point x
        self.nb_grad+=1
        return np.array([x[0],7*x[1]])
    def Hess(self,x) :
        # returns the Hessian of the function at point x
        self.nb_hess+=1
        to_return=np.zeros((2,2))
        to_return[0,0]=1
        to_return[1,1]=7
        return to_return
  

class Rosen() :
    def __init__(self) :
        print("Fonction (x,y) --> 100*(y-x^2)^2 + (1-x)^2")
        self.zeros()
    def zeros(self) :
        self.nb_eval=0
        self.nb_grad=0
        self.nb_hess=0
    def value(self,x) :
        self.nb_eval+=1
        return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    def grad(self,x) : 
        self.nb_grad+=1
        return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),200*(x[1]-x[0]**2)])
    def Hess(self,x) :
        self.nb_hess+=1
        to_return=np.zeros((2,2))
        to_return[0,0]=1200*x[0]**2-400*x[1]+2
        to_return[0,1]=-400*x[0]
        to_return[1,0]=-400*x[0]
        to_return[1,1]=200
        return to_return


class oscill() :
    def __init__(self) :
        print("Fonction (x,y) --> (1/2)x^2 + x*cos(y)")
        self.zeros()
    def zeros(self) :
        self.nb_eval=0
        self.nb_grad=0
        self.nb_hess=0
    def value(self,x) :
        self.nb_eval+=1
        return (1/2)*x[0]**2+x[0]*np.cos(x[1])
    def grad(self,x) :
        self.nb_grad+=1
        return np.array([x[0]+np.cos(x[1]),-x[0]*np.sin(x[1])])
    def Hess(self,x) :
        self.nb_hess+=1
        to_return=np.zeros((2,2))
        to_return[0,0]=1
        to_return[0,1]=-np.sin(x[1])
        to_return[1,0]=-np.sin(x[1])
        to_return[1,1]=-x[0]*np.cos(x[1])
        return to_return


