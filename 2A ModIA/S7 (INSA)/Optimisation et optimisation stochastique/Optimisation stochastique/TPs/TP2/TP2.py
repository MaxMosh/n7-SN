#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:36:26 2022

@author: pierre
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

## We need to decide whether we use the GPU or the CPU
use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu") 
if use_cuda :
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
print("GPU: ", use_cuda)

## To rescale an image in [0,1]
def rescale_01(y_t):
    y_t -= y_t.amin(dim = (2,3),keepdim=True)
    y_t /= y_t.amax(dim = (2,3),keepdim=True)
    return y_t

## Here we generate the random 
def generate_data(batch_size, size_image, sigma1, sigma2, sigma):
    # Defining a grid
    xs = torch.linspace(-size_image/2, size_image/2, size_image).type(dtype)
    ys = torch.linspace(-size_image/2, size_image/2, size_image).type(dtype)
    # The type of x,y below is automatically set to dtype, because xs, ys are of type dtype
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    
    # Defining 4 Gaussian
    g = torch.exp(-(x**2+y**2)/(2*sigma**2))
    g = g/torch.sum(g)

    g1 = torch.exp(-(x**2+y**2)/(2*sigma1**2))
    g1 = g1/torch.sum(g1)

    g2 = torch.exp(-(x**2+y**2)/(2*sigma2**2))
    g2 = g2/torch.sum(g2)

    g3 = torch.exp(-(x**2+y**2)/(2*1**2))
    g3 = g3/torch.sum(g3)
    
    # We define a random smooth partition
    b = torch.randn(batch_size,1,size_image,size_image).type(dtype)
    gp = F.conv2d(b, g[None,None], padding = 'same')
    omega1 = gp>0
    
    # We define two Gaussian random processes with differents statistics
    b1 = torch.randn(batch_size,1,size_image,size_image).type(dtype)
    gp1 = F.conv2d(b1, g1[None,None], padding = 'same')
    b2 = torch.randn(batch_size,1,size_image,size_image).type(dtype)
    gp2 = F.conv2d(b2, g2[None,None], padding = 'same')
    
    # Now we put each process at the right place
    u = gp1*omega1 + gp2*(~omega1)
    # We smooth the result to avoid having a jump at the interfaces
    u =  F.conv2d(u, g3[None,None], padding = 'same')
    u = rescale_01(u)
    
    return u, omega1.type(dtype)
    
## My first CNN
class my_first_CNN(nn.Module):
    def __init__(self,num_channels=16,bias=True):
        super(my_first_CNN,self).__init__()

        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_channels)
        self.bias = bias
        self.kernel_size = 5
        self.pad = int((self.kernel_size - 1) / 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=bias,
            ),
            self.activation,
            self.bn
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
            ),
            self.activation,
            self.bn
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
            ),
            self.activation,
            self.bn
        )

        self.out = nn.Conv2d(
                in_channels=num_channels,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
        )

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = torch.sigmoid(self.out(x3))
        return x

## The main script
batch_size = 10
size_image = 100
sigma1 = 1. # Parameter to describe Gaussian process 1
sigma2 = 2 # Parameter to describe Gaussian process 2
sigma = size_image/5. # Parameter to describe the image smoothness
learning_rate = 1e-3
display_it = 100
niter_train = 100000

## Defining the neural network
predictor = my_first_CNN(num_channels=16,bias=True).type(dtype)
predictor.to(device) # We decide whether the network operates on GPU or CPU
print(predictor) # We can display the model

## Just to see what generate_data does
u, omega1 = generate_data(batch_size, size_image, sigma1, sigma2, sigma)

if u.is_cuda : 
    u = u.cpu().detach().numpy()
    omega1 = omega1.cpu().detach().numpy()
    
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(u[0,0]) 
plt.subplot(1,2,2)
plt.imshow(omega1[0,0])

plt.imsave("image_processus.png",u[0,0])
plt.imsave("image_partition.png",omega1[0,0])
    
## Now, we are ready to train
optimizer = optim.Adam(predictor.parameters(), lr = learning_rate, betas=(0.9,0.999), eps=1e-8)
optimizer = optim.SGD(predictor.parameters(), lr = learning_rate, momentum = 0.9)
i = 0
avg_loss = 0
CF = []
while i<niter_train:
    i+=1
    # Generating the data at random
    x, omega1 = generate_data(batch_size, size_image, sigma1, sigma2, sigma)
    # Gradient based descent
    optimizer.zero_grad()
    prediction = predictor(x)    
    # Defining the loss function
    loss = torch.sum((prediction - omega1.type(dtype))**2)/batch_size
    # Computing the adjoint state (backprop)
    loss.backward()
    # One step of a gradient descent
    optimizer.step()
    
    CF.append(loss.item())
    avg_loss += loss.item()
    
    if i % display_it ==0 :
        plt.figure(1)
        plt.subplot(1,3,1)
        plt.imshow(x[0,0].cpu().detach().numpy())
        plt.title('Iteration %i/%i' % (i,niter_train))
        plt.subplot(1,3,2)
        plt.imshow(omega1[0,0].cpu().detach().numpy())
        plt.subplot(1,3,3)
        plt.imshow(prediction[0,0].cpu().detach().numpy())
        plt.title('CF %1.2e' % (avg_loss/display_it) )
        plt.show()
        
        plt.figure(2)
        plt.semilogy(CF)
        
        avg_loss = 0