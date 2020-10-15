'''
Model: Social Models
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

class Social_LSTM_abs(nn.Module):
    def __init__(self, data_loader, args, train = False):
        super(Social_LSTM_abs, self).__init__()

        #-----General Parameters
        self.use_cuda = args.use_cuda
        self.nmixtures = args.nmixtures
        self.rnn_size=  args.rnn_size 
        self.embedding_size = args.embedding_size                         # size of embedding layers for ALL 
        self.predict_length = args.predict_length  
        self.observe_length = args.observe_length
        self.tsteps = args.observe_length + args.predict_length            # number of steps 
        self.num_layers = args.num_layers                                  # number of layers for ALL LSTMs
        self.output_size = self.nmixtures*6                                # final output size 
        self.dropout = args.dropout if(train) else 0

        # Social Parameters 
        self.neighborhood_size = 0.4
        self.social_grid_size = args.social_grid_size

        #-----Layer 1: Individual Movements
        self.ReLU = nn.ReLU()
        self.I_Embedding = nn.Linear(2,self.embedding_size)     
        self.embedding_social = nn.Linear(self.social_grid_size*self.social_grid_size*self.rnn_size,self.embedding_size)
        self.I_LSTM      = nn.LSTM(self.embedding_size*2, self.rnn_size, num_layers=1, dropout=self.dropout)      
        self.I_Output     = nn.Linear(self.rnn_size, self.output_size)
    
    def init_batch_parameters(self, batch):
        """
            functions init all needed info for each batch
        """
        self.ped_ids = batch["all_pids"]                      # Set ped ids for current batch
        self.dataset_id = batch["dataset_id"]                # Set dataset id for current batch

    def init_target_hidden_states(self):
       # Initialize states for all targets in current batch
        self.i_h0 = torch.zeros(self.num_layers, len(self.ped_ids), self.rnn_size)
        self.i_c0 = torch.zeros(self.num_layers, len(self.ped_ids), self.rnn_size)
        if(self.use_cuda): 
            self.h0, self.c0 = self.h0.cuda(), self.c0.cuda() 

    def forward(self, xabs):

        # Embedded x,y coordinates with ReLU activation 
        #embedding_x ~ [1, batch_size, embbeding_size]        
        embedding_x = self.I_Embedding(xabs.unsqueeze(0))
        embedding_x = self.ReLU(embedding_x)

        # Find neighborhoods for all targets
        # grid_mask ~ [numPeds,numPeds,self.social_grid_size*self.social_grid_size] 
        grid_mask = self.getGridMask(xabs)

        # Calculate social tensor ~ [ped, self.social_grid_size**2*rnn_size]
        social_tensor = self.getSocialTensor(grid_mask)

        # hidden_pooling ~ [1,batch_size, self.embedding_size]
        hidden_pooling  = self.embedding_social(social_tensor).unsqueeze(0)
        hidden_pooling  = self.ReLU(hidden_pooling)

        # Concatenate input embediing and hidden pooling
        # lstm_input ~ [1,batch_size, 2*embedding_size]
        lstm_input = torch.cat([embedding_x,hidden_pooling],dim = 2)

        # I_LSTM_output ~ [1,batch_size,rnn_size]
        i_lstm_output,(self.i_h0, self.i_c0) = self.I_LSTM(lstm_input,(self.i_h0, self.i_c0))

        # I_output ~ [1,batch_size, embedding_size]
        final_output = self.I_Output(i_lstm_output)

        final_output = final_output.view(-1,self.output_size)

        # split output into pieces. Each ~ [1*batch_size, nmixtures]
        # and store them to final results
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = final_output.split(self.nmixtures,dim=1)          

        return mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits

    def process(self, cur_frame, next_frame):

        # Get input data of time t , x ~ [self.batch_size,2]
        xoff       = Variable(torch.from_numpy(cur_frame["loc_off"])).float()
        xoff_next  = Variable(torch.from_numpy(next_frame["loc_off"])).float()   
        if(self.use_cuda): xoff, xoff_next  = xoff.cuda(), xoff_next.cuda()     
        xabs      = Variable(torch.from_numpy(cur_frame["loc_abs"])).float()
        xabs_next = Variable(torch.from_numpy(next_frame["loc_abs"])).float()
        if(self.use_cuda): xabs, xabs_next  = xabs.cuda(), xabs_next.cuda()    

        self.batch_size = xabs.size(0)

        # forward 
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = self.forward(xabs)

        #loss_t = self.calculate_loss_mse(xabs_pred, xabs_next, cur_frame["frame_pids"], next_frame["frame_pids"])
        loss_t = self.calculate_loss(xabs_next, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, cur_frame["frame_pids"], next_frame["frame_pids"])
        
        return loss_t
    
    def calculate_loss(self, x_next, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, ped_ids_t, ped_ids_next):

        # x1, x2 ~ [self.batch_size, 1]   
        x_next  = x_next.view(-1,2) 
        x1, x2 = x_next.split(1,dim=1)

        def logP_gaussian(x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, nmixtures):
            
            x1, x2 = x1.repeat(1,nmixtures), x2.repeat(1,nmixtures)
            sigma1, sigma2 = log_sigma1.exp(), log_sigma2.exp()
            rho = torch.tanh(rho)
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

        loss = -logP_gaussian(x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, self.nmixtures)

        return loss/self.batch_size 

    def sample(self, cur_frame):

        # Get input data of time t , x ~ [1,self.batch_size,2]
        xoff  = Variable(torch.from_numpy(cur_frame["loc_off"])).float()
        xabs  = Variable(torch.from_numpy(cur_frame["loc_abs"])).float()          
        if(self.use_cuda): xoff, xabs  = xoff.cuda(), xabs.cuda()   

        self.batch_size = xabs.size(0)

        # forward 
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = self.forward(xabs)

        # re-calculate inputs for next prediction step
        xabs_pred = torch.cat([mu1.data,mu2.data], dim = 1)
        xoff_pred = xabs_pred - xabs

        return xoff_pred.cpu().data.numpy(), xabs_pred.cpu().data.numpy()

    def getGridMask(self, x_t):
    
        # x_t is location at time t 
        # x_t ~ [batch_size,2]
        frame_mask =  torch.zeros((self.batch_size, self.batch_size, self.social_grid_size**2))
        if(self.use_cuda): frame_mask = frame_mask.cuda()

        # For each ped in the frame (existent and non-existent)
        for pedindex in range(self.batch_size):

            # Get x and y of the current ped
            current_x, current_y = x_t[pedindex, 0].item(), x_t[pedindex, 1].item()

            width_low, width_high = current_x - self.neighborhood_size/2, current_x + self.neighborhood_size/2
            height_low, height_high = current_y - self.neighborhood_size/2, current_y + self.neighborhood_size/2

            # For all the other peds
            for otherpedindex in range(self.batch_size):

                # If the other pedID is the same as current pedID
                # The ped cannot be counted in his own grid
                if (otherpedindex == pedindex) :
                    continue

                # Get x and y of the other ped
                other_x, other_y = x_t[otherpedindex, 0].item(), x_t[otherpedindex, 1].item()
                if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                    # Ped not in surrounding, so binary mask should be zero
                    continue

                # If in surrounding, calculate the grid cell
                cell_x = int(np.floor(((other_x - width_low)/self.neighborhood_size) * self.social_grid_size))
                cell_y = int(np.floor(((other_y - height_low)/self.neighborhood_size) * self.social_grid_size))

                if cell_x >= self.social_grid_size or cell_x < 0 or cell_y >= self.social_grid_size or cell_y < 0:
                    continue

                # Other ped is in the corresponding grid cell of current ped
                frame_mask[pedindex, otherpedindex, cell_x + cell_y*self.social_grid_size] = 1

        return frame_mask


    def getSocialTensor(self, grid_mask):

        # grid_mask ~ [batch_size,batch_size,self.social_grid_size**2]
        # Construct the variable
        social_tensor = Variable(torch.zeros(self.batch_size, self.social_grid_size*self.social_grid_size, self.rnn_size))
        if(self.use_cuda): social_tensor = social_tensor.cuda()

        # For each ped, create social tensor (rnn_size) in each grid cell, 
        # if 2 other peds in same cell then h= h1 + h2, which 
        # can be done be followed matrix multiplation
        for ped in range(self.batch_size):
            # grid_mask[ped] ~ [batch_size, self.social_grid_size**2]
            # hidden_state ~ [1,batch_size,rnn_size]
            # social_tensor[ped]  ~ [self.social_grid_size**2, rnn_size]
            social_tensor[ped] = torch.mm(torch.t(grid_mask[ped]), self.i_h0.squeeze(0))

        # Reshape the social tensor
        social_tensor = social_tensor.view(self.batch_size, self.social_grid_size*self
            .social_grid_size*self.rnn_size)

        return social_tensor
    
    