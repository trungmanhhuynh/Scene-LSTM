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
from utils.scene_grids import get_sub_grid_cell_index

def logsumexp(x):
    x_max, _ = x.max(dim=1,keepdim=True)
    x_max_expand = x_max.expand(x.size())
    res =  x_max + torch.log((x-x_max_expand).exp().sum(dim=1, keepdim=True))
    return res

class Model_LSTM_Scene_common_nosubgrids(nn.Module):
    def __init__(self, data_loader, args, train = False):
        super(Model_LSTM_Scene_common_nosubgrids, self).__init__()

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
        self.nonlinear_grids =  args.nonlinear_grids

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
        self.F_gate = nn.Linear(self.rnn_size + self.embedding_size, self.rnn_size)
        self.Final_Output    = nn.Linear(self.rnn_size, self.output_size)

        ##Init## 
        initialize_scene_data(data_loader) 

    def init_batch_parameters(self, batch):
        """
            functions init all needed info for each batch
        """
        self.ped_ids = batch["ped_ids"]                      # Set ped ids for current batch
        self.dataset_id = batch["dataset_id"]                # Set dataset id for current batch
        self.num_peds = self.ped_ids.size                    # Set number of peds for current batch

        # Initialize states for all targets in current batch
        self.h0 = Variable(torch.zeros(self.num_layers, self.num_peds, self.rnn_size))
        self.c0 = Variable(torch.zeros(self.num_layers, self.num_peds, self.rnn_size))
        self.h0, self.c0 = self.h0.cuda(), self.c0.cuda() if(args.use_cuda)


    def get_target_hidden_states(self, batch_cur): 
        
        indices = np.where(batch_cur["frame_pids"][:, None] ==  self.ped_ids[None, :])[1]            
        indices = Variable(torch.LongTensor(indices))
        if(args.use_cuda): indices = indices.cuda()

        # Init hidden/cell states for peds
        self.i_h0 = torch.index_select(self.h0,1,indices).clone()
        self.i_c0 = torch.index_select(self.c0,1,indices).clone()

    def update_target_hidden_states(self, batch_cur): 
        
        indices = np.where(batch_cur["frame_pids"][:, None] ==  self.ped_ids[None, :])[1]            
        indices = Variable(torch.LongTensor(indices))
        if(args.use_cuda): indices = indices.cuda()

        self.h0[:,indices,:] = self.i_h0.clone()
        self.c0[:,indices,:] = self.i_c0.clone()

    def forward(self, x, xabs):
        # Forward bacth data at time t

        # Get one-hot vectors of locations
        # scene_embedding ~ [1, self.num_peds,inner_grid_size**2]
        scene_embedding = self.get_onehot_location(xabs)

        input_scene =  torch.cat([scene_embedding,self.i_h0], dim = 2)

        lstm_scene_output, (self.scene_grid_h0, self.scene_grid_c0) = self.LSTM_Scene(input_scene, (self.scene_grid_h0, self.scene_grid_c0))
        lstm_scene_output = self.ReLU(lstm_scene_output)

        embedding_input = self.Embedding_Input(x.unsqueeze(0))


        #embedding_input = self.ReLU(embedding_input)
        i_lstm_output_l1,(self.i_h0, self.i_c0) = self.I_LSTM_L1(embedding_input,(self.i_h0, self.i_c0))
        #i_lstm_output_l1 = self.ReLU(i_lstm_output_l1)

        # which information of the scene should be use to 
        # adjust individual movements ?  
        filter_input = torch.cat([embedding_input,i_lstm_output_l1], dim = 2)
        filter_gate =  self.F_gate(filter_input)
        filter_gate =  self.Sigmoid(filter_gate) 
        filtered_scene_lstm = filter_gate*lstm_scene_output             # element-wise multiplication

        # Modify the individual movements by filtered scene
        i_lstm_output_l1 = i_lstm_output_l1 + filtered_scene_lstm
        i_lstm_output_l1 = self.ReLU(i_lstm_output_l1)

        # Convert to final output 
        final_output = self.Final_Output(i_lstm_output_l1)

        final_output = final_output.view(-1,self.output_size)
        # split output into pieces. Each ~ [1*self.num_peds, nmixtures]
        # and store them to final results
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = final_output.split(self.nmixtures,dim=1)   

        return mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits

    def process(self, batch_cur, batch_next):

        # Get input data of time t , x ~ [1,self.num_peds,2]
        xoff, xoff_next = Variable(torch.from_numpy(batch_cur["loc_off"])), Variable(torch.from_numpy(batch_next["loc_off"]))          
        xoff, xoff_next  = xoff.cuda(), xoff_next.cuda()    if(args.use_cuda): 
        xabs, xabs_next = Variable(torch.from_numpy(batch_cur["loc_abs"])), Variable(torch.from_numpy(batch_next["loc_abs"]))
        xabs, xabs_next  = xabs.cuda(), xabs_next.cuda()    if(args.use_cuda)
        x = xoff if(self.predict_distance) else x = xabs 
        x_next = xoff_next if(self.predict_distance) else x_next = xabs_next 

        #  get individual and scene states for peds in current frame
        self.get_target_hidden_states(batch_cur)             # Get the hidden states of targets in current frames
        self.get_scene_states(xabs)                          # each ~ [num_layers,num_peds,rnn_size]

        # Each ~ [self.num_peds, nmixtures]
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits = self.forward(x, xabs)

        loss_t = self.calculate_loss(x_next, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, batch_cur["frame_pids"], batch_next["frame_pids"])

        # Update individual and scene states back to storage
        self.update_target_hidden_states(batch_cur)
        self.update_scene_states(xabs)

        return loss_t

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

        # x1, x2 ~ [self.num_peds, 1]   
        x_next  = x_next.view(-1,2) 
        x1, x2 = xoff_next.split(1,dim=1)
        x1  = torch.index_select(x1,0,indices_tplus1)
        x2  = torch.index_select(x2,0,indices_tplus1)

        def logP_gaussian(x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, nmixtures):
            x1, x2 = x1.repeat(1,nmixtures), x2.repeat(1,nmixtures)
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

        loss = logP_gaussian(x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, self.nmixtures)

        return loss/self.num_peds 

    def get_onehot_location(self, x):

        # x ~ [1,self.num_peds, 2]

        # Initialize one-hot location
        one_hot = Variable(torch.zeros(1, self.num_peds, self.num_sub_grids**2))
        if(self.use_cuda):
            one_hot = one_hot.cuda()
        
        # For each ped in the frame
        for pedindex in range(self.num_peds):

            # Get x and y of the current ped
            ped_location = np.array([0,0])
            ped_location[0], ped_location[1] = x[pedindex, 0].cpu().numpy(), x[pedindex, 1].cpu().numpy()

            # Get sub-grid index
            sub_grid_indx =  get_sub_grid_cell_index(ped_location, self.num_grid_cells, self.num_sub_grids, range =[-1,1,-1,1])

            one_hot[:,pedindex,sub_grid_indx] = 1 

        return one_hot

    def initialize_scene_data(self, data_loader):
        # Initialize scene states for all dataset.
        # Intilize the grid-cell memories of 2 types: non-linear and common movments.
        self.scene_states = {
            "nonlinear_h0_list": torch.zeros(args.num_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size),
            "nonlinear_c0_list": torch.zeros(args.num_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size),
            "common_h0_list" : torch.zeros(args.num_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size),
            "common_c0_list": torch.zeros(args.num_datasets,args.num_grid_cells**2,args.num_layers, 1, args.rnn_size)
        }
        if(self.use_cuda): 
            scene_states["nonlinear_h0_list"], = scene_states["nonlinear_h0_list"].cuda() 
            scene_states["nonlinear_c0_list"]  = scene_states["nonlinear_c0_list"].cuda()
            scene_states["common_h0_list"]     = scene_states["common_h0_list"].cuda()
            scene_states["common_c0_list"]     = scene_states["common_c0_list"].cuda()

        # Define scene information
        self.scene_info ={ 
            "nonlinear_grid_list": [] ,
            "nonlinear_sub_grids_maps" : [], 
            "common_grid_list" : [], 
            "common_sub_grids_maps" : []
        }
        if(self.num_common_grids > 0):
            self.scene_info["common_grid_list"], self.scene_info["common_sub_grids_maps"] = get_common_grids(data_loader, args)
        
        if(self.nonlinear_grids):
            self.scene_info["common_grid_list"], self.scene_info["common_sub_grids_maps"] = get_nonlinear_grids(data_loader, args)


    def get_scene_states(self, xabs):  
        """
        functions get scene_states (scene memories) for each current pedestrian
        """        
        # define 2 types (nonlinear and common movements) of a grid-cell memory
        nonlinear_grid_h0 =  torch.zeros((self.num_layers, self.num_peds, self.rnn_size))  
        nonlinear_grid_c0 =  torch.zeros((self.num_layers, self.num_peds, self.rnn_size)) 
        common_grid_h0 =  torch.zeros((self.num_layers, self.num_peds, self.rnn_size))
        common_grid_c0 =  torch.zeros((self.num_layers, self.num_peds, self.rnn_size))  
        scene_grid_h0 =  torch.zeros((self.num_layers, self.num_peds, self.rnn_size))
        scene_grid_c0 =  torch.zeros((self.num_layers, self.num_peds, self.rnn_size))  

        if(self.use_cuda):
            nonlinear_grid_h0, nonlinear_grid_c0 =  nonlinear_grid_h0.cuda(), nonlinear_grid_c0.cuda() 
            common_grid_h0, common_grid_c0 =  common_grid_h0.cuda(), common_grid_c0.cuda() 

        # For each ped in the frame (existent and non-existent)
        for pid in range(self.num_peds):

            gid =  get_grid_cell_index(xabs[pid].cpu().numpy(), self.num_grid_cells)
            sid =  get_sub_grid_cell_index(xabs[pid].cpu().numpy(), self.num_grid_cells, self.num_sub_grids)

            if(self.nonlinear_grids and gid in scene_info["nonlinear_grid_list"][self.dataset_id]):
                if((self.use_sub_grids_map and sid in scene_info["nonlinear_sub_grids_maps"][self.dataset_id][gid]) \
                    or self.use_sub_grids_map is False):

                    nonlinear_grid_h0[:,pid,:]  = self.scene_states["nonlinear_h0_list"][self.dataset_id][gid]
                    nonlinear_grid_c0[:,pid,:]  = self.scene_states["nonlinear_c0_list"][dataset_id][gid]

            if(self.num_common_grids >0 and gid in scene_info["common_grid_list"][self.dataset_id]): 
                if((self.use_sub_grids_map and sid in scene_info["common_sub_grids_maps"][self.dataset_id][gid]) \
                    or self.use_sub_grids_map is False ):

                    common_grid_h0[:,pid,:]  = self.scene_states["common_h0_list"][self.dataset_id][gid]
                    common_grid_c0[:,pid,:]  = self.scene_states["common_c0_list"][self.dataset_id][gid]

            # return the final scene states are average of nonlinear and common movements. 
            scene_grid_h0[:,pid,:] = 0.5*(nonlinear_grid_h0[:,pid,:] + common_grid_h0[:,pid,:])
            scene_grid_c0[:,pid,:] = 0.5*(nonlinear_grid_c0[:,pid,:] + common_grid_c0[:,pid,:])

        self.scene_grid_h0 = Variable(scene_grid_h0.view(-1,self.num_peds,self.rnn_size))
        self.scene_grid_c0 = Variable(scene_grid_c0.view(-1,self.num_peds,self.rnn_size))
        if(self.use_cuda):
            self.scene_grid_h0, self.scene_grid_c0 =  self.scene_grid_h0.cuda(), self.scene_grid_c0.cuda() 

    def update_scene_states(xabs):

        """
        functions update scene_states (scene memories)
        """
        for pid in range(self.num_peds):
            gid =  get_grid_cell_index(xabs[pid].cpu().numpy(), args.num_grid_cells)
            sid =  get_sub_grid_cell_index(xabs[pid].cpu().numpy(), args.num_grid_cells, args.num_sub_grids)

            if(args.nonlinear_grids and gid in scene_info["nonlinear_grid_list"][dataset_id]):
                if((args.use_sub_grids_map and sid in scene_info["nonlinear_sub_grids_maps"][dataset_id][gid]) \
                    or args.use_sub_grids_map is False):

                    scene_states["nonlinear_h0_list"][dataset_id][gid] = scene_grid_h0[:,pid,:].data.clone()
                    scene_states["nonlinear_c0_list"][dataset_id][gid] = scene_grid_c0[:,pid,:].data.clone()

            if(args.num_common_grids > 0 and gid in scene_info["common_grid_list"][dataset_id]): 
                if((args.use_sub_grids_map and sid in scene_info["common_sub_grids_maps"][dataset_id][gid]) \
                    or args.use_sub_grids_map is False ):      
                    
                    scene_states["common_h0_list"][dataset_id][gid] = scene_grid_h0[:,pid,:].data.clone()
                    scene_states["common_c0_list"][dataset_id][gid] = scene_grid_c0[:,pid,:].data.clone()


        return scene_states
