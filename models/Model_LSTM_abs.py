'''
 Model: Vanilla with location offsets as inputs.
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

def logsumexp(x):
    x_max, _ = x.max(dim=1,keepdim=True)
    x_max_expand = x_max.expand(x.size())
    res =  x_max + torch.log((x-x_max_expand).exp().sum(dim=1, keepdim=True))
    return res

def isnan(x):
    return x != x

class Model_LSTM_abs(nn.Module):
    def __init__(self, data_loader, args, train = False):
        super(Model_LSTM_abs, self).__init__()

        #-----General Parameters
        self.use_cuda = args.use_cuda
        self.nmixtures = args.nmixtures                                    # number of mixtures in final output
        self.rnn_size = args.rnn_size                                      # rnn_size of ALL LSTMs
        self.embedding_size = args.embedding_size                          # size of embedding layers for ALL 
        self.predict_length = args.predict_length  
        self.observe_length = args.observe_length
        self.tsteps = args.observe_length + args.predict_length            # number of steps 
        self.num_layers = args.num_layers                                  # number of layers for ALL LSTMs
        self.output_size = self.nmixtures*6                               # final output size 
        #self.output_size = 2                                               # final output size 
        self.dropout = args.dropout if(train) else 0

        #-----Layer 1: Individual Movements
        self.ReLU = nn.ReLU()
        self.Embedding_Input = nn.Linear(2,self.embedding_size)
        self.I_LSTM      = nn.LSTM(self.embedding_size, self.rnn_size, num_layers=1, dropout=self.dropout)      
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

    def update_target_hidden_states(self, cur_frame): 
        
        indices = np.where(cur_frame["frame_pids"][:, None] ==  self.ped_ids[None, :])[1]            
        indices = torch.LongTensor(indices)
        if(self.use_cuda): indices = indices.cuda()

        self.h0[:,indices,:] = self.i_h0.clone()
        self.c0[:,indices,:] = self.i_c0.clone()

    def forward(self, xabs):
        # Forward bacth data at time t

        embedding_input = self.Embedding_Input(xabs.unsqueeze(0))
        embedding_input = self.ReLU(embedding_input)
        # I_LSTM_output ~ [1,batch_size,rnn_size]
        i_lstm_output,(self.i_h0, self.i_c0) = self.I_LSTM(embedding_input,(self.i_h0, self.i_c0))
        i_lstm_output = self.ReLU(i_lstm_output)
        
        # I_output ~ [1,batch_size, embedding_size]
        final_output = self.I_Output(i_lstm_output)

        final_output = final_output.view(-1,self.output_size)

        # split output into pieces. Each ~ [1*batch_size, nmixtures]
        # and store them to final results
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = final_output.split(self.nmixtures,dim=1)          

        return mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits

        #return final_output

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

    def sample(self, cur_frame):

        # Get input data of time t , x ~ [1,self.batch_size,2]
        xoff  = Variable(torch.from_numpy(cur_frame["loc_off"])).float()
        xabs  = Variable(torch.from_numpy(cur_frame["loc_abs"])).float()          
        if(self.use_cuda): xoff, xabs  = xoff.cuda(), xabs.cuda()   

        self.batch_size = xoff.size(0)

        # forward 
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = self.forward(xabs)

        # re-calculate inputs for next prediction step
        xabs_pred = torch.cat([mu1.data,mu2.data], dim = 1)
        xoff_pred = xabs_pred- xabs

        return xoff_pred.cpu().data.numpy(), xabs_pred.cpu().data.numpy()

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


    def calculate_loss_mse(self, x_pred, x_next, ped_ids_t, ped_ids_next):

        # Which data should be used to calculate loss ? 
        # Only caculate for peds that presents in both frames 
        indices = np.where(ped_ids_t[:, None] == ped_ids_next[None, :])
        
        # if there is no same peds present in next frame
        # then return loss = 0 ?
        if(indices[0].size ==0): return 0 

        indices_t = torch.LongTensor(indices[0])
        indices_tplus1 = torch.LongTensor(indices[1])
        if(self.use_cuda):
            indices_t,indices_tplus1 = indices_t.cuda(), indices_tplus1.cuda()

        # x1, x2 ~ [self.batch_size, 1]   
        x_next  = x_next.view(-1,self.output_size) 
        x_next  = torch.index_select(x_next,0,indices_tplus1)
        x_pred  = torch.index_select(x_pred,0,indices_t)

        criterion = nn.MSELoss()
        loss_t = torch.sqrt(criterion(x_pred, x_next))
        
        return loss_t