import numpy as np
import scipy.linalg

def deriv_num(J,a,d,compute_grad=True,compute_Hess=True) :
    """test numerically the derivative and the Hessian of a function.
        
    Parameters
    ----------
    J : instance of a class
        The function to be tested it must have the following methods, where x is a 1d vector
        of size n
            -- J.eval(x) : evaluation of J at point x, must return a float
            -- J.grad(x) : evaluation of the gradient of J at point x, must a 1d vector of size n
            -- J.Hess(x) : evaluation of the Hessian of J at point x, typically a n*n matrix
    a : 1d vector of size n
        Point at which the numerical derivatives are evaluated
    d : 1d vector of size n
        Direction in which the numerical derivatives are evaluated
    compute_grad : Boolean
        Flag that tests the function J.grad against numerical derivatives
    compute_Hess : Boolean
        Flag that tests the function J.Hess against numerical derivatives of J.grad
    
   Ouput 
   -----
   This function does not have an output, it prints a string s.
    """
        
    eps_range=[0.1**(i+1) for i in range(12)]
    for eps in  eps_range:
        s='eps {:1.3e}'.format(eps)
        if compute_grad :
            # ratio of numerical derivatives of J and prediction given by J.grad
            ratio=(J.value(a+eps*d)-J.value(a))/(eps*np.dot(J.grad(a),d)) 
            s+=' grad {:1.1e}'.format(np.abs(ratio-1)) 
        if compute_Hess :
            v1=(J.grad(a+eps*d)-J.grad(a))/eps # numerical derivative of J.grad
            v2=J.Hess(a).dot(d)  # prediction given by J.Hess
            ratio=np.linalg.norm(v1)/np.linalg.norm(v2) #norm ratio
            angle=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) # cosinus of the angle between the vectors
            s+=' ratio {:1.1e}'.format(np.abs(ratio-1.))
            s+=' angle {:1.1e}'.format(angle-1.)
        print(s)
        
def main_algorithm(function,step,xini,dc,ls,itermax = 20000,tol=1.e-4,verbose=True):
    """Perform a minimization algorithm of a function
    
    Parameters
    ----------
    function : instance of a class
        The function to be minimized, depending on the choice of linesearch and direction of descent,
        it must have the following methods, where x is a 1d vector of size n
            -- function.eval(x) : evaluation of J at point x, must return a float
            -- function.grad(x) : evaluation of the gradient of J at point x, must a 1d vector of size n
            -- function.Hess(x) : evaluation of the Hessian of J at point x, typically a n*n matrix
    step : positive float
        Initial guess of the step
    xini : 1d vector of size n
        initial starting point
    dc : callable
        descent,info_dc=dc(x,function,df,res)
        computes the descent direction with parameters 
           -x: the point x
           -df : the gradient of function at point x
           -function : the function
        The function dc returns
            -descent : the direction of descent
            -info_dc : information about the behavior of the function dc   
    ls : callable
        x2,f2,df2,step2,info_ls=ls(x, function, step, descent,f,df)
        performs a line search, the parameters are
           -x : initial point
           -step : initial step
           -function : the function to be minimized
           -f,df : the values of function(x) and the gradient of the function of x
           -descent : the descent direction
        the function returns
            -x2 : the new point x+step2*descent
            -f2 : the value of function at point x2
            -df2 : the value of the gradient of the function at point x2
            -step2 : the step given by the linesearch
            -info_ls : some information about the behavior of the function ls
    itermax : int
        maximum number of iterations
    tol : float
       stopping criterion
    verbose : Boolean
        Printing option of the algorithm
    
    Returns
    --------
    The function returns a single dictionnary res, the entries are
    res['list_x'] : list of 1d vectors of size n which are the iterates points of the algorithm
    res['list_steps'] : list of positive floats which are the different steps
    res['list_grads'] : list of positive floats which are the value of the euclidean norm of the gradients of the function
    res['final_x'] : 1d vector, final value of x
    res['dc'] : list of the different infos given by the functions dc
    res['ls'] : list of the different infos given by the functions ls        
    """
    x = xini
    res={'list_x':[],'list_steps':[],'list_costs':[],'list_grads':[],'final_x':[],'dc':[],'ls':[]}
    nbiter = 0
    f=function.value(x)
    df= function.grad(x)
    err=np.linalg.norm(df)
    if verbose :  print('iter={:4d} f={:1.3e} df={:1.3e} comp=[{:4d},{:4d},{:4d}]'.format(nbiter,f,err,function.nb_eval,function.nb_grad,function.nb_hess))
    res['list_x'].append(x.copy())
    res['list_costs'].append(f)
    res['list_grads'].append(err)
    while (err > tol) and (nbiter < itermax):
        descent,info_dc = dc(x, function,df)
        x,f,df,step,info_ls = ls(x, function, step, descent,f,df)
        err = np.linalg.norm(df)
        res['list_x'].append(x.copy())
        res['list_costs'].append(f)
        res['list_grads'].append(err)
        res['list_steps'].append(step)
        res['dc'].append(info_dc)
        res['ls'].append(info_ls)
        nbiter+=1
        if verbose : print('iter={:4d} f={:1.3e} df={:1.3e} comp=[{:4d},{:4d},{:4d}]'.format(nbiter,f,err,function.nb_eval,function.nb_grad,function.nb_hess))
        if (err <= tol):
            res['final_x']=np.copy(x)
            if verbose : print("Success !!! Algorithm converged !!!")
            return res
    if verbose : print("FAILED to converge")
    return res        

def dc_gradient(x,function,df) :
    """Choice of direction of descent : GRADIENT METHOD
    
    Parameters
    ----------
        x : 1d vector of size n
            actual iterate of the method
        function : instance of a class
            The function to be minimized
        df : The actual value of the gradient
          
    returns
    -------
       descent : 1d vector of size n
           direction of descent 
       ls_info : 
           Information about the behavior of the function
    """
    descent=-df
    ls_info=None
    return descent,ls_info

def ls_constant(x, function, step, descent,f,df) :
    """Line search : FIXED STEP
    
    Parameters
    ----------
        x : 1d vector of size n
            actual iterate of the method
        function : instance of a class
            The function to be minimized
        step : float
            The starting guess of the step
        descent : 1d vector of size n
            The descent direction
        f : float
            the value of the function at point x
        df : 1d vector of size n
            the gradient of the function at point x
          
    returns
    -------
        x2 : 1d vector of size n
            x2=x+step2*descent
        f2 : float
            the value of the function at point x2
        df2 : 1d vector of size n
            the gradient of the function at point x2
        step2 : float
            The step chosen by the method
        info : Information about the behavior of the method 
       
    """
    step2=step
    x2=x+step2*descent
    f2=function.value(x2)
    df2=function.grad(x2)
    info=None
    return x2,f2,df2,step2,info


#x2,f2,df2,step2,info = ls_backtracking(x, function, step, descent,f,df)
def ls_backtracking(x, function, step, descent, f, df) :
    """Line search : VARIABLE STEP
    
    Parameters
    ----------
        x : 1d vector of size n
            actual iterate of the method
        function : instance of a class
            The function to be minimized
        step : float
            The starting guess of the step
        descent : 1d vector of size n
            The descent direction
        f : float
            the value of the function at point x
        df : 1d vector of size n
            the gradient of the function at point x
          
    returns
    -------
        x2 : 1d vector of size n
            x2=x+step2*descent
        f2 : float
            the value of the function at point x2
        df2 : 1d vector of size n
            the gradient of the function at point x2
        step2 : float
            The step chosen by the method
        info : Information about the behavior of the method 
       
    """

    
    step2=step
    x2=x+step2*descent
    f2=function.value(x2)
    

    while f2 > f :
        step2=step2/2
        x2=x+step2*descent
        f2=function.value(x2)
    

    info=None
    df2=function.grad(x2)
    return x2,f2,df2,step2,info


def ls_partial_linesearch(x, function, step, descent, f, df) :

    stepList = [0.1*step,0.5*step,step,2*step,10*step]
    fList = [function.value(x+s*descent) for s in stepList]

    f2 =  min(fList)
    step2 = stepList[np.argmin(fList)]

    x2 = x + step2*descent
    df2 = function.grad(x2)
    info = None

    return x2,f2,df2,step2,info

""" Fait par mes soins en recopiant le prof
def ls_wolfe(x, function, step, descent, f, df) :
    step_min, step_max = 0.0, np.inf
    i = 0
    mycontinue = True
    scal = np.dot(df,descent)
    step2 = step
    eps1, eps2 = 0.1, 0.9
    while mycontinue and (i<100) :
        i = i + 1
        mycontinue = False
        x2 = x + step*descent
        f2 = function.value(x2)
        if f2 > f + eps1*step2*scal :
            # step is too big, decrease it
            step_max = step2
            step2 = 0.5*(step_min + step_max)
            mycontinue = True
        else :
            df2 = function.grad(x2)
            if np.dot(df2,descent) < eps2*scal :
                #step is too small, increase it
                step_min = step2
                step2 = min(0.5*(step_min + step_max), 2*step_min)
                mycontinue = True
    return x2, f2, df2, step2, None
"""



def ls_wolfe(x, function, step, descent,f,df):
    step_min = 0.0
    step_max = np.inf
    i=0
    mycontinue = True
    scal=np.dot(df,descent)
    step2=step
    eps1,eps2 = 0.1 , 0.9
    while mycontinue and (i<100) :
        i=i+1
        mycontinue = False
        x2=x+step2*descent
        f2=function.value(x2)
        if f2 > f+eps1*step2*scal:
            step_max=step2
            step2=0.5*(step_min+step_max)
            mycontinue = True
        else:
            df2=function.grad(x2)
            if np.dot(df2,descent) <eps2*scal :
                step_min=step2  
                step2=min(0.5*(step_min+step_max),2*step_min)
                mycontinue = True
    return x2,f2,df2,step2,None






"""

À FAIRE
def ls_wolfe(x, function, step, descent, f, df) :

    eps1 = 10**(-4)
    eps2 = 0.9

    borneInf = 0
    borneSup = np.inf

    continue = True

    step2=step

    while continue :
            
        if function.value(x + step2*descent) <= function.value(x) + eps1*step2*np.transpose(function.grad(x))*descent :
            borneSup = step2
        elif np.transpose(function.grad(x + step2*descent))*descent >= eps2*np.transpose(function.grad(x))*descent :
            borneInf = step2
        else :
            continue = False
        step2 = (borneInf + borneSup)/2


    x2 = x + step*descent
    f2 = function.value(x2)
    df2 = function.grad(x2)
    
    info=None


    
    
    x2=x+step2*descent
    f2=function.value(x2)
    df2=function.grad(x2)
    


    return x2,f2,df2,step2,info
    """



def dc_newton(x, function,df):
    Hf =function.Hess(x)
    descent = np.linalg.solve(Hf,-df)
    angle = np.dot(descent,df)/(np.linalg.norm(descent)*np.linalg.norm(df))
    print(angle)
    if angle < -0.1:
        return descent , "newton"
    else :
        return -df , "gradient"


def ls_wolfe_step_is_one(x,function,step,descent,f,df) :
    return ls_wolfe(x,function,1.,descent,f,df)


class BFGS() :
    def __init__(self,nb_stock_max=8) :
        self.nb_stock_max=nb_stock_max
        self.stock=[]
        self.last_iter=None
    def push(self, x, grad) :
        xk = x
        gradk = grad
        if self.last_iter is not None :
            xkmoinsun, gradkmoinsun = self.last_iter
            sigmak = xk - xkmoinsun
            yk = gradk - gradkmoinsun
            rhok = 1/(np.dot(sigmak,yk))
            if rhok >= 0 :
                if len(self.stock) >= self.nb_stock_max :
                    self.stock.pop(0)
                else :
                    pass
                self.stock.append((np.copy(sigmak),np.copy(yk),np.copy(rhok)))
            else :
                self.stock.clear()
            self.last_iter = (np.copy(xk),np.copy(gradk))
        else :
            self.last_iter = (np.copy(x),np.copy(grad))
    def get(self, grad) :
        if len(self.stock) == 0 :
            return -grad
        else :
            q = - np.copy(grad)
            listealpha = []
            for i in range(len(self.stock)) :
                sigmai, yi, rhoi = self.stock[-i-1]
                sigmakmin, ykmin, rhokmin = self.stock[0]
                alphai = rhoi * np.dot(sigmai,q)
                listealpha.insert(0, alphai)
                q = q - alphai*yi
            q = (np.dot(sigmakmin,ykmin)/np.dot(ykmin,ykmin))*q
            for i in range(len(self.stock)) :
                sigmai, yi, rhoi = self.stock[i]
                alphai = listealpha[i]
                betai = rhoi * np.dot(yi, q)
                q = q + (alphai - betai)*sigmai
        return q
    def dc(self,x,function,df) :
        self.push(x, df)
        descent = self.get(df)
        return descent, None