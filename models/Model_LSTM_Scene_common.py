'''
 Scene Model
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
from utils.scene_grids import get_nonlinear_grids, get_common_grids, get_grid_cell_index, get_sub_grid_cell_index

def logsumexp(x):
    x_max, _ = x.max(dim=1,keepdim=True)
    x_max_expand = x_max.expand(x.size())
    res =  x_max + torch.log((x-x_max_expand).exp().sum(dim=1, keepdim=True))
    return res

def isnan(x):
    return x != x

class Model_LSTM_Scene_common(nn.Module):
    def __init__(self, data_loader, args, train = False):
        super(Model_LSTM_Scene_common, self).__init__()

        #-----General Parameters
        self.use_cuda = args.use_cuda
        self.nmixtures = args.nmixtures                                    # number of mixtures in final output
        self.rnn_size = args.rnn_size                                      # rnn_size of ALL LSTMs
        self.embedding_size = args.embedding_size                          # size of embedding layers for ALL 
        self.predict_length = args.predict_length  
        self.observe_length = args.observe_length
        self.tsteps = args.observe_length + args.predict_length            # number of steps 
        self.num_layers = args.num_layers                                  # number of layers for ALL LSTMs
        self.output_size = self.nmixtures*6                                # final output size 
        self.dropout = args.dropout if(train) else 0
        self.predict_distance = args.predict_distance                      # set train/test flag

        #-----Scene Parameters    
        self.num_grid_cells = args.num_grid_cells                          # number of grid size for scene
        self.num_sub_grids = args.num_sub_grids
        self.scene_mixtures = args.scene_mixtures 
        self.scene_output_size = self.scene_mixtures*6                      # number of grid size for scene
        self.num_common_grids =  args.num_common_grids

        #---- Scene Models
        self.Sigmoid = nn.Sigmoid() 
        self.ReLU = nn.ReLU() 
        self.Tanh = nn.Tanh()

        # convert absolute location to one-hot locations 
        self.LSTM_Scene = nn.LSTM(self.num_sub_grids**2 + self.rnn_size, self.rnn_size, num_layers=1, dropout=self.dropout)

        #--- Individual model
        self.Embedding_Input = nn.Linear(2,self.embedding_size)
        self.I_LSTM_L1    = nn.LSTM(self.embedding_size, self.rnn_size, num_layers=1, dropout=self.dropout)

        # Modify individual LSTM
        self.Soft_Gate = nn.Linear(self.rnn_size + self.num_sub_grids**2, self.rnn_size)
        self.Final_Output    = nn.Linear(self.rnn_size, self.output_size)

        ##Init## 
        self.initialize_scene_data(data_loader, args) 

    def init_batch_parameters(self, batch):
        """
            functions init all needed info for each batch
        """
        self.ped_ids = batch["all_pids"]                      # Set ped ids for current batch
        self.dataset_id = batch["dataset_id"]                # Set dataset id for current batch

        # Initialize states for all targets in current batch
        self.h0 = torch.zeros(self.num_layers, self.ped_ids.size, self.rnn_size)
        self.c0 = torch.zeros(self.num_layers, self.ped_ids.size, self.rnn_size)
        if(self.use_cuda): 
            self.h0, self.c0 = self.h0.cuda(), self.c0.cuda() 

    def get_target_hidden_states(self, cur_frame): 
        
        indices = np.where(cur_frame["frame_pids"][:, None] ==  self.ped_ids[None, :])[1]            
        indices = torch.LongTensor(indices)
        if(self.use_cuda): indices = indices.cuda()

        # Init hidden/cell states for peds
        self.i_h0 = torch.index_select(self.h0,1,indices).clone()
        self.i_c0 = torch.index_select(self.c0,1,indices).clone()

    def update_target_hidden_states(self, cur_frame): 
        
        indices = np.where(cur_frame["frame_pids"][:, None] ==  self.ped_ids[None, :])[1]            
        indices = torch.LongTensor(indices)
        if(self.use_cuda): indices = indices.cuda()

        self.h0[:,indices,:] = self.i_h0.clone()
        self.c0[:,indices,:] = self.i_c0.clone()

    def forward(self, x, xabs):
        # Forward bacth data at time t

        # Scene LSTM 
        scene_embedding = self.get_onehot_location(xabs)            #  ~ [1, self.batch_size, sub_grids_size**2]
        input_scene =  torch.cat([scene_embedding, self.i_h0], dim = 2)
        lstm_scene_output, (self.scene_grid_h0, self.scene_grid_c0) = self.LSTM_Scene(input_scene, (self.scene_grid_h0, self.scene_grid_c0))
        lstm_scene_output = self.ReLU(lstm_scene_output)

        # Individual LSTM
        embedding_input = self.Embedding_Input(x.unsqueeze(0))
        i_lstm_output,(self.i_h0, self.i_c0) = self.I_LSTM_L1(embedding_input,(self.i_h0, self.i_c0))
        i_lstm_output = self.ReLU(i_lstm_output)

        # Soft Filter
        filter_input = torch.cat([scene_embedding, i_lstm_output], dim = 2)
        soft_gate =  self.Soft_Gate(filter_input)
        soft_filter =  self.Sigmoid(soft_gate) 
        filtered_scene_lstm = soft_filter*lstm_scene_output             # element-wise multiplication

        # Modify the individual movements by filtered scene
        i_lstm_output = i_lstm_output + filtered_scene_lstm
        i_lstm_output = self.ReLU(i_lstm_output)

        # Convert to final output 
        final_output = self.Final_Output(i_lstm_output)

        final_output = final_output.view(-1,self.output_size)

        # split output into pieces. Each ~ [1*self.batch_size, nmixtures]
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = final_output.split(self.nmixtures,dim=1)   

        return mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits

    def process(self, cur_frame, next_frame):

        # Get input data of time t , x ~ [self.batch_size,2]
        xoff       = Variable(torch.from_numpy(cur_frame["loc_off"]), requires_grad= True ).float()
        xoff_next  = Variable(torch.from_numpy(next_frame["loc_off"])).float()   
        if(self.use_cuda): xoff, xoff_next  = xoff.cuda(), xoff_next.cuda()     
        xabs      = Variable(torch.from_numpy(cur_frame["loc_abs"]), requires_grad= True).float()
        xabs_next = Variable(torch.from_numpy(next_frame["loc_abs"])).float()
        if(self.use_cuda): xabs, xabs_next  = xabs.cuda(), xabs_next.cuda()    
        x = xoff if(self.predict_distance) else xabs 
        x_next = xoff_next if(self.predict_distance) else xabs_next 

        self.batch_size = xabs.size(0)

        #  get individual and scene states for peds in current frame
        self.get_target_hidden_states(cur_frame)             # Get the hidden states of targets in current frames
        self.get_scene_states(xabs)                          # each ~ [num_layers,batch_size,rnn_size]

        # forward 
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = self.forward(x, xabs)
        if(isnan(mu1[0])):
            print("mu1 = ", mu1)
            print("mu2 = ", mu2)
            print(" xoff =" , xoff)
            print(" xabs =" , xabs)
            input("here")

        loss_t = self.calculate_loss(x_next, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, cur_frame["frame_pids"], next_frame["frame_pids"])

        # update individual and scene states back to storage
        self.update_target_hidden_states(cur_frame)
        self.update_scene_states(xabs)

        return loss_t

    def sample(self, cur_frame):

        # Get input data of time t , x ~ [1,self.batch_size,2]
        xoff  = Variable(torch.from_numpy(cur_frame["loc_off"])).float()
        xabs  = Variable(torch.from_numpy(cur_frame["loc_abs"])).float()          
        if(self.use_cuda): xoff, xabs  = xoff.cuda(), xabs.cuda()   
        x = xoff if(self.predict_distance) else xabs 
        self.batch_size = xabs.size(0)

        #  get individual and scene states for peds in current frame
        self.get_target_hidden_states(cur_frame)             # Get the hidden states of targets in current frames
        self.get_scene_states(xabs)                          # each ~ [num_layers,batch_size,rnn_size]

        # forward 
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = self.forward(x, xabs)

        #Re-calculate inputs for next prediction step
        xoff = torch.cat([mu1.data,mu2.data], dim = 1)
        xabs_pred = xabs + xoff
        xabs_pred.cpu().data.numpy()
        return xoff.cpu().data.numpy(), xabs_pred.cpu().data.numpy()


    def calculate_loss(self, x_next, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, ped_ids_t, ped_ids_next):

        # Which data should be used to calculate loss ? 
        # Only caculate for peds that presents in both frames 
        indices = np.where(ped_ids_t[:, None] == ped_ids_next[None, :])
        
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

        # x1, x2 ~ [self.batch_size, 1]   
        x_next  = x_next.view(-1,2) 
        x1, x2 = x_next.split(1,dim=1)
        x1  = torch.index_select(x1,0,indices_tplus1)
        x2  = torch.index_select(x2,0,indices_tplus1)

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

        loss = logP_gaussian(x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, self.nmixtures)

        return loss/self.batch_size 

    def get_onehot_location(self, x):

        # x ~ [1,self.batch_size, 2]

        # Initialize one-hot location
        one_hot = Variable(torch.zeros(1, self.batch_size, self.num_sub_grids**2))
        if(self.use_cuda):
            one_hot = one_hot.cuda()
        
        # For each ped in the frame
        for pedindex in range(self.batch_size):

            # Get x and y of the current ped
            ped_location = np.array([0,0])
            ped_location[0], ped_location[1] = x[pedindex, 0].detach().numpy(), x[pedindex, 1].detach().numpy()

            # Get sub-grid index
            sub_grid_indx =  get_sub_grid_cell_index(ped_location, self.num_grid_cells, self.num_sub_grids, range =[-1,1,-1,1])

            one_hot[:,pedindex,sub_grid_indx] = 1 

        return one_hot

    def initialize_scene_data(self, data_loader, args):
        
        # Initialize scene states for all dataset.
        self.scene_states = {
            "common_h0_list" : torch.zeros(args.max_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size),
            "common_c0_list": torch.zeros(args.max_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size)
        }
        if(self.use_cuda): 
            scene_states["common_h0_list"]     = scene_states["common_h0_list"].cuda()
            scene_states["common_c0_list"]     = scene_states["common_c0_list"].cuda()

        # Define scene information
        self.scene_info ={ 
            "common_grid_list" : [], 
        }
        # Get common movements info
        print("get common grids...")
        self.scene_info["common_grid_list"], _ = get_common_grids(data_loader, args)
        print(self.scene_info["common_grid_list"])

    def get_scene_states(self, xabs):  
        """
        functions get scene_states (scene memories) for each current pedestrian
        """        

        common_grid_h0 =  torch.zeros((self.num_layers, self.batch_size, self.rnn_size))
        common_grid_c0 =  torch.zeros((self.num_layers, self.batch_size, self.rnn_size))  
        if(self.use_cuda):
            common_grid_h0, common_grid_c0 =  common_grid_h0.cuda(), common_grid_c0.cuda() 

        for pid in range(self.batch_size):
            # For each ped in current frame
            gid =  get_grid_cell_index(xabs[pid].detach().numpy(), self.num_grid_cells)        # current grid-cell
            if(gid in self.scene_info["common_grid_list"][self.dataset_id]): 
                # if current grid-cells is in common grid list then extract it to use
                common_grid_h0[:,pid,:]  = self.scene_states["common_h0_list"][self.dataset_id][gid]
                common_grid_c0[:,pid,:]  = self.scene_states["common_c0_list"][self.dataset_id][gid]

        self.scene_grid_h0 = common_grid_h0.view(-1,self.batch_size,self.rnn_size)
        self.scene_grid_c0 = common_grid_c0.view(-1,self.batch_size,self.rnn_size)
        if(self.use_cuda):
            self.scene_grid_h0, self.scene_grid_c0 =  self.scene_grid_h0.cuda(), self.scene_grid_c0.cuda() 

    def update_scene_states(self, xabs):

        """
        functions update scene_states (scene memories)
        """
        for pid in range(self.batch_size):
            # For each ped in current frame
            gid =  get_grid_cell_index(xabs[pid].detach().numpy(), self.num_grid_cells) # current grid-cell
            if(gid in self.scene_info["common_grid_list"][self.dataset_id]): 
                # if grid-cell is in common_grid_list then update
                self.scene_states["common_h0_list"][self.dataset_id][gid] = self.scene_grid_h0[:,pid,:].data.clone()
                self.scene_states["common_c0_list"][self.dataset_id][gid] = self.scene_grid_c0[:,pid,:].data.clone()