
import matplotlib.pyplot as plt
import numpy as np
import torch

#from lin2d_exp import x_dim, b_size, m, sigma0

from filters import Lin2d
from filters import ConstructorProp
from filters import ConstructorObs

import manage_exp
import lin2d_exp

T = 50

Lin2dExp = Lin2d(lin2d_exp.x_dim, N=1, dt=0, init="0")
#liste_t = [t for t in range(50)]
liste_x = []

#x = Lin2dExp.x0                                                     # x0 si on change pas mb
x = manage_exp.get_x0(2,lin2d_exp.x_dim,lin2d_exp.sigma0)      # si on change mb
for t in range(T):
    x = Lin2dExp.forward(x)

    liste_x.append(x)

#plt.plot(liste_t, liste_x)
plt.plot([x[0,0] for x in liste_x], [x[0,1] for x in liste_x], 'bo')
plt.plot([x[1,0] for x in liste_x], [x[1,1] for x in liste_x], 'ro')
plt.show()



x0 = manage_exp.get_x0(2,lin2d_exp.x_dim,lin2d_exp.sigma0)      # si on change mb

exp_params = lin2d_exp.get_params()

prop_params = exp_params["prop_kwargs"]
obs_params = exp_params["obs_kwargs"]

prop = ConstructorProp(**prop_params)
obs = ConstructorObs(**obs_params)

xtps = torch.empty(T,lin2d_exp.b_size,lin2d_exp.x_dim)

yts = torch.empty(T,lin2d_exp.b_size,lin2d_exp.x_dim)

xtp = x0

for t in range(T):
    xtp = prop(xtp).sample()
    yt = obs(xtp).sample()
    xtps[t] = xtp
    yts[t] = yt

print(yts)
plt.plot([x[0,0] for x in liste_x], [x[0,1] for x in liste_x], 'go', label='Solution de référence')
plt.plot([y[0,0] for y in yts], [y[0,1] for y in yts], 'bo', label='Solution obtenue par propagation')
plt.plot([x[0,0] for x in xtps], [x[0,1] for x in xtps], 'ro', label='Solution obtenue par observation')
plt.legend()
plt.show()
plt.plot([x[1,0] for x in liste_x], [x[1,1] for x in liste_x], 'go', label='Solution de référence')
plt.plot([y[1,0] for y in yts], [y[1,1] for y in yts], 'bo', label='Solution obtenue par propagation')
plt.plot([x[1,0] for x in xtps], [x[1,1] for x in xtps], 'ro', label='Solution obtenue par observation')
plt.legend()
plt.show()