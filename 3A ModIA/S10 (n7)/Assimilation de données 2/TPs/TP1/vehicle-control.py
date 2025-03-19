#
# A simple shooting method to solve a dynamic control problem
#
# Direct model: the basic 1D trajectory of vehicle dy/dt(t) = tau*(-y(t) + G * u(t)) + IC.
#     State of the sytem: vehicle velocity y(t) (ms-1)
#     Control parameter: pedal position u(t)
#     G and tau are constant parameters
#
# Objective function minimized: J(u;y)= int (y - y_target)^2 + alpha du/dt^2 dt, on a given future time intervall 
#
# Th present code is an enriched / adapted version of a code  available  on-line, from  Ass. Prof. Hedengren, J. D., USA. 
# Version enriched by Prof. J. Monnier and INSA students, France.
#
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize

#
# Define the state equation
#
def direct_model(y,t,u,G,tau):
    # compute dy/dt(t) in a given time intervall, given the parameters
    # arguments
    #  y   = the system state
    #  t   = current time
    #  u   = the control var.
    #  G   = gain paramater (=ratio of u)
    #  tau = K/m, time parameter

    # the ode expression 
    dydt = tau * (-y + G * u)

    return dydt 

#
# Define the objective function      
#
def objective(u_hat):
    # Compute the objective function in the prediction intervall [P+1:2*P+1]    
    # To do so, the ODE has to be solved. 
    # Subscript _hat denotes variables defined in the current time intervall [i-P:i+P],
    #      intervall centered on i with i = current time loop index
    #
    # Argument:
    #     u_hat: the control variable in the current time intervall t_hat only
    #
    # Recall:
    # P: prediction horizon index; M: control horizon index; M<=P. 
    #
    
    # loop on time instants
    for k in range(1,2*P+1): # k varies in t_hat, the current working intervall
        if k==1:
            y_hat0 = yp[i-P] # IC relatively to the current working intervall
            
        # Copy of the global values of the previous values of u (estimation obtained at the previous instant i) into u_hat
        # Moreover, u is extrapolated for instants farer than (P+M)
        if k<=P: # 1st half of the intervall
            if i+k-P<0:
                u_hat[k] = 0 # out left of the intervall
            else:  
                u_hat[k] = u[i+k-P] # u_hat = u at the correct index
        elif k>P+M: # the non controlled part of the 2nd half of the intervall
            u_hat[k] = u_hat[P+M] # the latest (P+M)-th value is extended
            
        # CAUTION: Values of u_hat[P+1:P+M] are those provided by the minimizer routine (see main program).
        # => do not assign these values...
        # Below, Jobj is computed by employing the previous values of u
        
        # Solve the ODE between the two current control times indices [(k-1),k]
        y_hat = odeint(direct_model,y_hat0,[(k-1),k],args=(u_hat[k],G,tau))
        
        y_hat0 = y_hat[-1] # update the local IC
        yp_hat[k] = y_hat[0] # the new estimation of state y at current index k

        # Objective fct = misfit term + regularization (penalization) term 
        ytarget_hat[k] = ytarget[i]
        if k>P: # 2nd half of intervall = prediction intervall
            Jmis_val[k] = (ytarget_hat[k]-yp_hat[k])**2 # misfit term at time index k
            # definition of the regularization term
            #Jreg_val[k] = (u_hat[k]-u_hat[k-1])**2 # MAY BE CHANGED
            Jreg_val[k] = u_hat[k]**2 
            # total cost function
            Jobj_val[k] = Jmis_val[k] + alpha_du * Jreg_val[k] # total cost function term at time index k 
    
    # The objective fct terms = sum of all terms in the future i.e. in intervall [P+1:2*P+1]
    Jmis = np.sum(Jmis_val[P+1:])
    Jreg = np.sum(Jreg_val[P+1:])
    # The total objective fct 
    Jobj = np.sum(Jobj_val[P+1:])
    
    return Jobj 


#
# To make MP4 animation
#
make_mp4 = False
if make_mp4:
    import imageio # required to make animation
    import os
    try:
        os.mkdir('./figures')
    except:
        pass
    
###############
# Main program
###############

# The model parameters
print('* Ready to make control ? *')
G = 3.0     # gain
tau = 1.e-2   # time constant
print ('Model parameters: G=',G,' tau=',tau)
Tfin = int(100.); print ('Final time: Tfin =', Tfin)
ns = 100 ; print ('Number of time steps: ns =', ns)
t = np.linspace(0,ns,Tfin) # the time grid to solve the ODE
delta_t = t[1]-t[0] # time step (in s)

# Initialization of the control var. & the state var. in the complete time intervall
u = np.zeros(ns+1)
yp = np.zeros(ns+1)

# The calibration (learning) time period & prediction time period
P = 20; print('The prediction intervall length (index unit) P=',P)
M = 20; print('The control horizon index  M=',M,'. One must have: M <= P')

# Regularization parameter
alpha_du = 0.01; print('The regularization weight parameter: alpha_du=',alpha_du) # MAY BE CHANGED

# additional inequality constraint on du (optional)
du_max = 1.; #print('Max variation of u between two sucessive time instants =',du_max)

# Define the target: ytarget(t)
ytarget = np.zeros(ns+P)
tar_val = [0.,5.,10.,3.,7.]
ytarget[0:round(ns/10.)] = tar_val[0]
ytarget[round(ns/10.):round(ns/4.)] = tar_val[1]
ytarget[round(ns/4.):round(ns/2.)] = tar_val[2]
ytarget[round(ns/2.):round(3*ns/4.)] = tar_val[3]
ytarget[round(3*ns/4.):] = tar_val[4]
    
#  Create plots
plt.figure(figsize=(10,6))
plt.ion()
plt.show()
# plot the target
plt.clf()
plt.subplot(2,1,1)
plt.plot(t[0:ns],ytarget[0:ns],'r-',linewidth=2,label='Target')
plt.axis([0, ns+M, 0, 12])
plt.xlabel('time t'); plt.ylabel('y_target (t) ')
plt.legend()
input('* Press return to resume ....')
plt.close()

#
# Loop in time steps: dynamics control policy
#

for i in range(1,ns):
    # working time index intervall
    # intervall centered on i, of lenght 2P. index i = present time
    t_hat = np.linspace(i-P,i+P,2*P+1) # t_hat contains past & future instant indices
    # print out
    if np.mod(i,10) == 0:
        print('* Current time index #',i,'. Recall: new optimization at each time step...')
        print('   Working index intervall = 2nd half of t_hat = [',t_hat[P+1],',...,',t_hat[-1],']')
        if i>=M: print('   Indices of u which are actually optimized:[',i+1,'...',i+M,']')
        
    if i==1:
        y0 = 0. # Initial Condition
    Deltat_i = [delta_t*(i-1),delta_t*i] # time intervall 
    y = odeint(direct_model,y0,Deltat_i,args=(u[i],G,tau)) # solve the state eqn in Deltat_i only
    y0 = y[-1] # update the local IC
    yp[i] = y[0] # store the newly computed state value at instant i

    # Setup & Initialize the _hat variables
    # subscript  _hat denotes variables defined on t_hat
    Jobj_val = np.zeros(2*P+1) # objective fct
    Jmis_val = np.zeros(2*P+1) # misfit term of Jobj_val    
    Jreg_val = np.zeros(2*P+1) # regularization term of Jobj_val    
    yp_hat = np.zeros(2*P+1) # state
    u_hat0 = np.zeros(2*P+1) # control first guess
    ytarget_hat = np.zeros(2*P+1) # target
    Jobj = 0.; Jmis=0.; Jreg=0. # objective/cost fct values

    # First guess of u (defined in the current working intervall)
    for k in range(1,2*P+1): # index varying in t_hat
        if k<=P:
            if i-P+k<0: # out of the global time intervall
                u_hat0[k] = 0. 
            else: # 1st pat of the intervall
                u_hat0[k] = u[i-P+k] # u_hat0 = u
        elif k>P: # 2nd part of the intervall
            u_hat0[k] = u[i] # extension of the lastest value previously estimated (constant extension)

    # print the resulting initial objective fct
    if np.mod(i,10) == 0:
        print('   Initial objective fct Jobj='+ str(objective(u_hat0)) )
#    print('       misfit  Jmis=' + str(objective(u_hat0)) )
#    print('       reg     Jreg=' + str(objective(u_hat0)) ) 

    #
    # Solve the optimization pb for the current prediction intervall 
    #
    startchrono = time.time() # start chrono
    
    # Minimization of Jobj( u([P+1]...[2P+1]) ) wrt to u in the current prediction intervall
    # Method: SLSQP = Sequential Least Squares Programming
    solution = minimize(objective, u_hat0, method='SLSQP')
    # NB. fun objective contains the ODE resolution
    # no additional constraint on u eg lower and upper bounds
    
    u_hat = solution.x  # the estimated optimal control for the current prediction intervall
    
    endchrono = time.time() # end chrono
    time_optim = endchrono - startchrono # CPU time 
    print('   CPU time of the optimization algo: ' + str(time_optim))
          
    # print the resulting objective fct
    if np.mod(i,10) == 0:
        print('   After optimization J_obj=' + str(objective(u_hat)))
    
    # Update the control value: shooting method, time instant by time instant...
    delta_u = np.diff(u_hat) # variations of u_hat
    if i<ns:
        u[i+1] = u[i]+delta_u[P] # futures values of u are adjusted by the newly computed estimation
    # Control policy (optional): variations of u are bounded to du_max if necessary
    if i>ns: # To be CHANGED IF one applies a bounding policy
        if np.abs(delta_u[P]) >= du_max: # if the variation of u at current time .ge. du_max, we bound
            if delta_u[P] > 0:
                u[i+1] = u[i]+du_max
            else:
                u[i+1] = u[i]-du_max
        else:
            u[i+1] = u[i]+delta_u[P] # future values of u are adjusted by the newly computed estimation 

    # plot the fields in the considered time intervall (= partial past+ partial future)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(t[0:i+1],ytarget[0:i+1],'r-',linewidth=2,label='Target')
    plt.plot(t_hat[P:],ytarget_hat[P:],'r--',linewidth=2)
    plt.plot(t[0:i+1],yp[0:i+1],'k-',linewidth=2,label='Obtained state y (in the past)')
    plt.plot(t_hat[P:],yp_hat[P:],'k--',linewidth=2,label='Predicted state y, given the estimated values of u')
    
    plt.axvline(x=i,color='gray',alpha=0.5) # in gray: the current time index
    plt.axis([0, ns+P, 0, 12])
    plt.ylabel('y(t)')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.step(t[0:i+1],u[0:i+1],'b-',linewidth=2,label='Pedal position: applied past values')
    plt.plot(t_hat[P:],u_hat[P:],'b--',linewidth=2,label='Estimated and extrapolated values')
    
    plt.axvline(x=i,color='gray',alpha=0.5)
    plt.axvline(x=i+M,color='green',alpha=0.2) # in green: the control horizon
    plt.ylabel('u(t)')
    plt.xlabel('time')
    plt.axis([0, ns+P, 0, 10])
    plt.legend()
    plt.draw()
    plt.pause(0.1)
    
    if make_mp4: # movie
        filename='./figures/plot_'+str(i+10000)+'.png'
        plt.savefig(filename)
    #
    if i == (M):
        print('***');print('!! It was piece of cake untill now: nothing had to be done ! Serious control begins now.')
        print('Recall. The prediction horizon index P=',P,'. The control horizon index M=',M)
        input('* Press return to resume ...')
        
    if np.mod(i,30) == 0:
        input('* Let us make a pause, press return to resume...'); print('*')

    if i == (ns-1):
        input('* That s all folks. Press return to close all.')
    
# end of time loop
        
#
# Generate mp4 from png figures in batches of 350
#
if make_mp4:
    images = []
    iset = 0
    for i in range(1,ns):
        filename='./figures/plot_'+str(i+10000)+'.png'
        images.append(imageio.imread(filename))
        if ((i+1)%350)==0:
            imageio.mimsave('results_'+str(iset)+'.mp4', images)
            iset += 1
            images = []
    if images!=[]:
        imageio.mimsave('results_'+str(iset)+'.mp4', images)
