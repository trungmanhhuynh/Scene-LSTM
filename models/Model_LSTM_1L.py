'''
Model: Vanilla-LSTM with embedding layers
Author: Huynh Manh

'''

import random
import math
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup as Soup
import numpy as np
import glob
from grid import *

def logsumexp(x):
    x_max, _ = x.max(dim=1,keepdim=True)
    x_max_expand = x_max.expand(x.size())
    res =  x_max + torch.log((x-x_max_expand).exp().sum(dim=1, keepdim=True))
    return res

class Model_LSTM_1L(nn.Module):
    def __init__(self, args, train = False):
        super(Model_LSTM_1L, self).__init__()

        #-----General Parameters
        self.use_cuda = args.use_cuda
        self.nmixtures = args.nmixtures
        self.rnn_size= 128
        self.predict_length = args.predict_length  
        self.observe_length = args.observe_length
        self.tsteps = args.observe_length + args.predict_length            # number of steps 
        self.num_layers = 1                                                # number of layers for ALL LSTMs
        self.embedding_size = 64                                           # size of embedding layers for ALL 
        self.output_size = self.nmixtures*6                                # final output size 
        self.train = train                                                 # set train/test flag
        self.predict_distance = args.predict_distance
        self.batch_size = 0         # I will set it for each frame

        if(train): 
            self.dropout = args.dropout 
        else: 
            self.dropout = 0       # set dropout value
        
        #-----Layer 1: Individual Movements
        self.ReLU = nn.ReLU()
        self.I_Embedding = nn.Linear(2,self.embedding_size)     
        self.I_LSTM      = nn.LSTM(self.embedding_size, self.rnn_size, num_layers=1, dropout=self.dropout)      
        self.I_Output     = nn.Linear(self.rnn_size, self.output_size)
    
    def set_target_hidden_states(self, h0_t, c0_t): 
        
        # Init hidden/cell states for Layer 1: 
        self.i_h0 = h0_t.clone()
        self.i_c0 = c0_t.clone()

    def forward(self, xoff, xabs):

        # Embedded x,y coordinates with ReLU activation 
        #embedding_x ~ [1, batch_size, embbeding_size]
        if(self.predict_distance):
            embedding_x = self.I_Embedding(xoff.unsqueeze(0))
        else:
            embedding_x = self.I_Embedding(xabs.unsqueeze(0))

        embedding_x = self.ReLU(embedding_x)

        # I_LSTM_output ~ [1,batch_size,rnn_size]
        i_lstm_output,(self.i_h0, self.i_c0) = self.I_LSTM(embedding_x,(self.i_h0, self.i_c0))

        # I_output ~ [1,batch_size, embedding_size]
        final_output = self.I_Output(i_lstm_output)

        final_output = final_output.view(-1,self.output_size)

        # split output into pieces. Each ~ [1*batch_size, nmixtures]
        # and store them to final results
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = final_output.split(self.nmixtures,dim=1)          

        return mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits

    def calculate_loss(self, xoff, xoff_next, xabs, xabs_next, ped_ids_t, ped_ids_tplus1):

         # Each ~ [batch_size, nmixtures]
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = self.forward(xoff, xabs)

        # Which data should be used to calculate loss ? 
        # Only caculate for peds that presents in both frames 
        indices = np.where(ped_ids_t[:, None] == ped_ids_tplus1[None, :])
        
        # if there is no same peds present in next frame
        # then return loss = 0 ?
        if(indices[0].size ==0): return 0 

        indices_t = Variable(torch.LongTensor(indices[0]))
        indices_tplus1 = Variable(torch.LongTensor(indices[1]))
        if(self.use_cuda):
            indices_t,indices_tplus1 = indices_t.cuda(), indices_tplus1.cuda()

        # Use indices to select which peds's location used for calculating loss
        mu1  = torch.index_select(mu1,0,indices_t)
        mu2  = torch.index_select(mu2,0,indices_t)
        log_sigma1  = torch.index_select(log_sigma1,0,indices_t)
        log_sigma2  = torch.index_select(log_sigma2,0,indices_t)
        rho  = torch.index_select(pi_logits,0,indices_t)
        pi_logits  = torch.index_select(pi_logits,0,indices_t)

        # x1, x2 ~ [batch_size, 1]   
        xabs_next, xoff_next  = xabs_next.view(-1,2) , xoff_next.view(-1,2) 
        if(self.predict_distance):
            x1, x2 = xoff_next.split(1,dim=1)
        else:
            x1, x2 = xabs_next.split(1,dim=1)

        x1  = torch.index_select(x1,0,indices_tplus1)
        x2  = torch.index_select(x2,0,indices_tplus1)

        loss = - self.logP_gaussian(x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits)

        return loss/self.batch_size 


    def logP_gaussian(self, x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits):

        x1, x2 = x1.repeat(1,self.nmixtures), x2.repeat(1,self.nmixtures)
        sigma1, sigma2 = log_sigma1.exp(), log_sigma2.exp()
        rho = nn.functional.tanh(rho)
        log_pi = nn.functional.log_softmax(pi_logits, dim = 1)
        z_tmp1, z_tmp2 = (x1-mu1)/sigma1, (x2-mu2)/sigma2
        z = z_tmp1**2 + z_tmp2**2 - 2*rho*z_tmp1*z_tmp2
        # part one
        log_gaussian = - math.log(math.pi*2)-log_sigma1 - log_sigma2 - 0.5*(1-rho**2).log()
        # part two
        log_gaussian += - z/2/(1-rho**2)
        # part three
        log_gaussian = logsumexp(log_gaussian + log_pi)

        return log_gaussian.sum()
    