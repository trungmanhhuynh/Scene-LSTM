'''
 Model: Model_LSTM_Scene_nonlinear_subgrids

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
from utils.scene_grids import get_nonlinear_grids, get_grid_cell_index, get_sub_grid_cell_index

def logsumexp(x):
    x_max, _ = x.max(dim=1,keepdim=True)
    x_max_expand = x_max.expand(x.size())
    res =  x_max + torch.log((x-x_max_expand).exp().sum(dim=1, keepdim=True))
    return res

def isnan(x):
    return x != x

class Model_LSTM_Scene_nonlinear_subgrids_noSF(nn.Module):
    def __init__(self, data_loader, args, train = False):
        super(Model_LSTM_Scene_nonlinear_subgrids_noSF, self).__init__()

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
        self.dropout = args.dropout if(train) else 0

        #-----Scene Parameters    
        self.num_grid_cells = args.num_grid_cells                          # number of grid size for scene
        self.num_sub_grids = args.num_sub_grids
        self.scene_mixtures = args.scene_mixtures 
        self.scene_output_size = self.scene_mixtures*6                      # number of grid size for scene

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
        self.Final_Output    = nn.Linear(self.rnn_size, self.output_size)

        ##Init## 
        if(train): self.initialize_scene_data(data_loader, args) 

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

    def forward(self, xoff, xabs):
        # Forward bacth data at time t

        # Scene LSTM 
        scene_embedding = self.get_onehot_location(xabs)            #  ~ [1, self.batch_size, sub_grids_size**2]
        input_scene =  torch.cat([scene_embedding, self.i_h0], dim = 2)
        lstm_scene_output, (self.scene_grid_h0, self.scene_grid_c0) = self.LSTM_Scene(input_scene, (self.scene_grid_h0, self.scene_grid_c0))
        lstm_scene_output = self.ReLU(lstm_scene_output)

        # Individual LSTM
        embedding_input = self.Embedding_Input(xoff.unsqueeze(0))
        embedding_input = self.ReLU(embedding_input)
        i_lstm_output,(self.i_h0, self.i_c0) = self.I_LSTM_L1(embedding_input,(self.i_h0, self.i_c0))
        i_lstm_output = self.ReLU(i_lstm_output)

        # Modify the individual movements by filtered scene
        final_lstm_output = i_lstm_output + lstm_scene_output
        final_lstm_output = self.ReLU(final_lstm_output)

        # Convert to final output 
        final_output = self.Final_Output(final_lstm_output)

        final_output = final_output.view(-1,self.output_size)

        # split output into pieces. Each ~ [1*self.batch_size, nmixtures]
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

        # set batch_size (number of peds in current frame)
        self.batch_size = xabs.size(0)

        #  get individual and scene states for peds in current frame
        self.get_scene_states(xabs)                          # each ~ [num_layers,batch_size,rnn_size]

        # forward 
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = self.forward(xoff, xabs)
        loss_t = self.calculate_loss(xoff_next, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, cur_frame["frame_pids"], next_frame["frame_pids"])
        
        # update individual and scene states back to storage
        self.update_scene_states(xabs)

        return loss_t

    def sample(self, cur_frame):

        # Get input data of time t , x ~ [1,self.batch_size,2]
        xoff  = Variable(torch.from_numpy(cur_frame["loc_off"])).float()
        xabs  = Variable(torch.from_numpy(cur_frame["loc_abs"])).float()          
        if(self.use_cuda): xoff, xabs  = xoff.cuda(), xabs.cuda()   

        self.batch_size = xoff.size(0)

        # forward 
        assert self.scene_states["nonlinear_h0_list"][self.dataset_id].sum() != 0, "Error: Zero scene data"
        assert self.scene_states["nonlinear_c0_list"][self.dataset_id].sum() != 0, "Error: Zero scene data"
        self.get_scene_states(xabs)                          # each ~ [num_layers,batch_size,rnn_size]
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = self.forward(xoff, xabs)

        # re-calculate inputs for next prediction step
        xoff_pred = torch.cat([mu1.data,mu2.data], dim = 1)
        xabs_pred = xabs + xoff_pred

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

    def set_scene_data(self, scene_states, scene_info):

        self.scene_states = scene_states
        self.scene_info = scene_info

    def initialize_scene_data(self, data_loader, args):
        
        # Initialize scene states for all dataset.
        self.scene_states = {
            "nonlinear_h0_list" : torch.zeros(args.max_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size),
            "nonlinear_c0_list" : torch.zeros(args.max_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size)
        }
        if(self.use_cuda): 
            self.scene_states["nonlinear_h0_list"]     = self.scene_states["nonlinear_h0_list"].cuda()
            self.scene_states["nonlinear_c0_list"]     = self.scene_states["nonlinear_c0_list"].cuda()

        # Define scene information
        self.scene_info ={ 
            "nonlinear_grid_list" : [], 
            "nonlinear_subgrid_maps": []
        }
        # Get nonlinear movements info     
        self.scene_info["nonlinear_grid_list"], self.scene_info["nonlinear_subgrid_maps"] = get_nonlinear_grids(data_loader, args)
        print(self.scene_info["nonlinear_grid_list"])
        print(self.scene_info["nonlinear_subgrid_maps"])

    def get_scene_states(self, xabs):  
        """
        functions get scene_states (scene memories) for each current pedestrian
        """        
        nonlinear_grid_h0 =  torch.zeros((self.num_layers, self.batch_size, self.rnn_size))
        nonlinear_grid_c0 =  torch.zeros((self.num_layers, self.batch_size, self.rnn_size))  
        if(self.use_cuda):
            nonlinear_grid_h0, nonlinear_grid_c0 =  nonlinear_grid_h0.cuda(), nonlinear_grid_c0.cuda() 

        for pid in range(self.batch_size):
            # For each ped in current frame
            gid =  get_grid_cell_index(xabs[pid].detach().numpy(), self.num_grid_cells)        # current grid-cell
            sid =  get_sub_grid_cell_index(xabs[pid].detach().numpy(), self.num_grid_cells, self.num_sub_grids)        # current grid-cell
            if(gid in self.scene_info["nonlinear_grid_list"][self.dataset_id] 
                and sid in self.scene_info["nonlinear_subgrid_maps"][self.dataset_id][gid]): 
                # if current grid-cells is in nonlinear grid list then extract it to use
                nonlinear_grid_h0[:,pid,:]  = self.scene_states["nonlinear_h0_list"][self.dataset_id][gid]
                nonlinear_grid_c0[:,pid,:]  = self.scene_states["nonlinear_c0_list"][self.dataset_id][gid]

        self.scene_grid_h0 = nonlinear_grid_h0.view(-1,self.batch_size,self.rnn_size)
        self.scene_grid_c0 = nonlinear_grid_c0.view(-1,self.batch_size,self.rnn_size)
        if(self.use_cuda):
            self.scene_grid_h0, self.scene_grid_c0 =  self.scene_grid_h0.cuda(), self.scene_grid_c0.cuda() 

    def update_scene_states(self, xabs):

        """
        functions update scene_states (scene memories)
        """
        for pid in range(self.batch_size):
            # For each ped in current frame
            gid =  get_grid_cell_index(xabs[pid].detach().numpy(), self.num_grid_cells) # current grid-cell
            sid =  get_sub_grid_cell_index(xabs[pid].detach().numpy(), self.num_grid_cells, self.num_sub_grids)        # current grid-cell

            if(gid in self.scene_info["nonlinear_grid_list"][self.dataset_id] 
                    and sid in self.scene_info["nonlinear_subgrid_maps"][self.dataset_id][gid]): 
                # if grid-cell is in nonlinear_grid_list then update
                self.scene_states["nonlinear_h0_list"][self.dataset_id][gid] = self.scene_grid_h0[:,pid,:].data.clone()
                self.scene_states["nonlinear_c0_list"][self.dataset_id][gid] = self.scene_grid_c0[:,pid,:].data.clone()

