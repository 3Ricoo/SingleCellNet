import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import GetData
import pandas as pd
import validation as val

(exp_data,init_weight,energy_data) = GetData.getSco2() 
#get GUO_data: GetData.getGuo()
(numGene, numData)= (np.shape(exp_data)[1],np.shape(exp_data)[0])


from torch.nn import Module, Parameter
###############  Define a cell in RNN ###########
class ODECell(Module):
    def __init__(self, n_inputs, n_neurons,h):
        super(ODECell, self).__init__()
        self.h = h
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.Wx = Parameter(torch.Tensor(n_inputs, n_neurons))
        ### self.Wx: Weight is a gene network, represented by a matrix;
        self.diag = Parameter(torch.Tensor(n_inputs))
        ### self.diag:A diagnal matrix; 
        self.bias = Parameter(torch.Tensor(1,n_inputs))
        self.const =  Parameter(torch.Tensor(1,n_inputs))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.n_neurons)
        for weight in self.parameters():
            weight.data.uniform_(-0.001, 0.001)
    def forward(self,X0):
        ###### Define ODE: ######
        '''
        TO DO: Selecting appropriate ODE
        Alternative Form:
        1. ODE in Hopland
        2. Hill function [refer paper: Reconstructing gene regulatory dynamics from high-dimensional single-cell snapshot data; Andrea Ocone; file-path: papers/grn-inder/ruleODE]
        3. Other classical equation in system biology and grn infer
        '''
        Y0 = torch.tanh(torch.mm(X0, self.Wx)-torch.mm(X0,torch.diag(self.diag)))
        ### 
        Y1 = X0+self.h*Y0
        return Y1

################ Define RNN ###################
import math
class ODERNN(nn.Module):
    def __init__(self, n_inputs, n_neurons,h):
        super(ODERNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.h = h
        ### self.h: time interval in numerial ODE
        self.cell = ODECell(n_inputs, n_neurons,h)
    def reward(self,x):
        '''
        TO DO: Design a reward function
        1. import env_V1.py
        2. reduce dimension: 
            2.1 use env_V1.pca_select
            2.2 other nonliner dimension reduction method
        3. calculating density/energy/distance
             - use env_V1.constructEnv 
        '''
        x1 = x.detach().numpy()
        x1 = x1[0]
        rew = (np.dot(x1,x1.T)+sum(x1))/100
        return rew
    def forward(self, X0,steps):
        out = torch.empty(steps,self.n_inputs)
        nn.init.zeros_(out)
        out[0] = X0
        rew = np.zeros(steps)
        rew[0]=0
        '''
        TO DO:
        rew = self.reward(X0)
        '''
        for i in range(steps-1):
            X0 = self.cell(X0)
            rew[i+1]= self.reward(X0)
            out[i+1] = X0
        return (out,rew)


from torch.autograd import Variable
odernn = ODERNN(numGene,numGene,1)
#### initialize parameters #####
odernn.cell.Wx = Parameter(torch.from_numpy(init_weight).float())
odernn.cell.diag = Parameter(torch.from_numpy(np.full((numGene),6)).float())

EPOCH = 50
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(odernn.parameters(),lr=0.001)
inputs = torch.tensor([exp_data[0]],dtype = torch.float)
datas = Variable(torch.from_numpy(exp_data))
datas = datas.float()

############### Training #####################
for epoch in range(EPOCH):    
    optimizer.zero_grad() 
    output,rew= odernn(inputs,numData)
    re = np.array([sum(rew)])
    loss = criterion(output,datas)
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients
    if((epoch+1)%50==0):
        print(epoch,"  loss  ",loss.item())

############# validation ##################
network = odernn.cell.Wx
out = abs(network).detach().numpy()
val.scode2_result(out,9000,100)