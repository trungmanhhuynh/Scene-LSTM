'''
Helper functions to compute the masks relevant to social grid

Author : Anirudh Vemula
Date : 29th October 2016
'''
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def getGridMask(x_t, neighborhood_size, grid_size):
    
    # x_t is location at time t 
    # x_t ~ [batch_size,2]

    # Maximum number of pedestrians
    num_peds = x_t.size(0)
    frame_mask =  Variable(torch.zeros((num_peds, num_peds, grid_size**2))).cuda()

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(num_peds):

        # Get x and y of the current ped
        current_x, current_y = x_t[pedindex, 0].item(), x_t[pedindex, 1].item()

        width_low, width_high = current_x - neighborhood_size/2, current_x + neighborhood_size/2
        height_low, height_high = current_y - neighborhood_size/2, current_y + neighborhood_size/2

        # For all the other peds
        for otherpedindex in range(num_peds):

            # If the other pedID is the same as current pedID
            # The ped cannot be counted in his own grid
            if (otherpedindex == pedindex) :
                continue

            # Get x and y of the other ped
            other_x, other_y = x_t[otherpedindex, 0].data[0], x_t[otherpedindex, 1].data[0]
            if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                # Ped not in surrounding, so binary mask should be zero
                continue

            # If in surrounding, calculate the grid cell
            cell_x = int(np.floor(((other_x - width_low)/neighborhood_size) * grid_size))
            cell_y = int(np.floor(((other_y - height_low)/neighborhood_size) * grid_size))

            if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue

            # Other ped is in the corresponding grid cell of current ped
            frame_mask[pedindex, otherpedindex, cell_x + cell_y*grid_size] = 1

    return frame_mask


def get_scene_states(x_t, dataset_id, scene_grid_num, scene_h0_list, scene_c0_list, args):
    
    # x_t is location at time t 
    # x_t ~ [batch_size,2]
    # scene_h0_list is list of grid hidden states
    # scene_c0_list is list of grid cell states 

    # Maximum number of pedestrians
    num_peds = x_t.size(0)
    rnn_size = scene_h0_list.size(4) # ~ [num_datasets,grid_Size**2,num_layers,batch_size,rnn_Size]
    num_layers = scene_h0_list.size(2)
    scene_grid_c0 =  Variable(torch.zeros((num_layers,num_peds, rnn_size)))  
    scene_grid_h0 =  Variable(torch.zeros((num_layers,num_peds, rnn_size)))  
    list_of_grid_id = Variable(torch.LongTensor(num_peds))
    if(args.use_cuda):
        list_of_grid_id, scene_grid_h0, scene_grid_c0 = list_of_grid_id.cuda(), scene_grid_h0.cuda(), scene_grid_c0.cuda() 

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(num_peds):

        # Get x and y of the current ped
        current_x, current_y = x_t[pedindex, 0].item(), x_t[pedindex, 1].item()

        width_low, width_high = -1 , 1          #scene is in range [-1,1]
        height_low, height_high = -1 , 1        #scene is in range [-1,1]
        boundary_size = 2 
  
        # calculate the grid cell
        cell_x = int(np.floor(((current_x - width_low)/boundary_size) * scene_grid_num))
        cell_y = int(np.floor(((current_y - height_low)/boundary_size) * scene_grid_num))

        # Peds locations must be in range of [-1,1], so the cell used must be in range [0,scene_grid_num-1]
        if(cell_x < 0):
            cell_x = 0
        if(cell_x >= scene_grid_num):
            cell_x = scene_grid_num - 1
        if(cell_y < 0):
            cell_y = 0 
        if(cell_y >= scene_grid_num):
            cell_y = scene_grid_num - 1

        list_of_grid_id[pedindex] = cell_x + cell_y*scene_grid_num

    scene_grid_h0 = torch.index_select(scene_h0_list[dataset_id],0,list_of_grid_id)
    scene_grid_c0 = torch.index_select(scene_c0_list[dataset_id],0,list_of_grid_id)

    scene_grid_c0 = scene_grid_c0.view(-1,num_peds,rnn_size)
    scene_grid_h0 = scene_grid_c0.view(-1,num_peds,rnn_size)

    return scene_grid_h0, scene_grid_c0, list_of_grid_id

def getGridMaskInference(frame, dimensions, neighborhood_size, grid_size):
    mnp = frame.shape[0]
    width, height = dimensions[0], dimensions[1]

    frame_mask = np.zeros((mnp, mnp, grid_size**2))

    width_bound, height_bound = (neighborhood_size/(width*1.0))*2, (neighborhood_size/(height*1.0))*2

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(mnp):
        # Get x and y of the current ped
        current_x, current_y = frame[pedindex, 0], frame[pedindex, 1]

        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2

        # For all the other peds
        for otherpedindex in range(mnp):
            # If the other pedID is the same as current pedID
            if otherpedindex == pedindex:
                # The ped cannot be counted in his own grid
                continue

            # Get x and y of the other ped
            other_x, other_y = frame[otherpedindex, 0], frame[otherpedindex, 1]
            if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                # Ped not in surrounding, so binary mask should be zero
                continue

            # If in surrounding, calculate the grid cell
            cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
            cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))

            if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue
            
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[pedindex, otherpedindex, cell_x + cell_y*grid_size] = 1

    return frame_mask

def getSequenceGridMask(sequence, dimensions, neighborhood_size, grid_size, using_cuda):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
        # sequence_mask[i, :, :, :] = getGridMask(sequence[i, :, :], dimensions, neighborhood_size, grid_size)
        mask = Variable(torch.from_numpy(getGridMask(sequence[i], dimensions, neighborhood_size, grid_size)).float())
        if using_cuda:
            mask = mask.cuda()
        sequence_mask.append(mask)

    return sequence_mask

def is_nonlinear(traj):
        non_linear_degree = 0

        y_max = traj[-1,1]
        y_min = traj[0,1]
        y_s   =traj[int(traj.shape[0]/2),1]
        if((y_max-y_min)/2 + y_min == 0):
            non_linear_degree = 0
        else:
            non_linear_degree = abs(((y_max - y_min)/2 + y_min - y_s))

        if(non_linear_degree >= 0.2): 
            return True
        else: 
            return False

def get_nonlinear_trajectories(batch,args):

    nonlinear_traj = []
    # Get trajectories for each frame
    for ped in batch["ped_ids"]: 
        traj = [] 
        for t in range(0,args.observe_length + args.predict_length):
            pid_idx = np.where(batch["ped_ids_frame"][t] == ped)[0]
            traj.append(batch["batch_data_absolute"][t][pid_idx])
        
        traj = np.vstack(traj[:])
        if(traj.shape[0] <= 5):
            continue

        # Check if this traj is non-linear 
        if(is_nonlinear(traj)): 
            nonlinear_traj.append(traj)

    return nonlinear_traj

def get_non_linear_grids(data_loader, args):

    allow_grid_list = [[] for i in range(5)]

    # DEFAULT: use all grids
    if(args.non_grids):
        # still empty 
        return allow_grid_list

    elif(args.nonlinear_grids):

        allow_grid_list[0] = [36, 37, 38, 39, 42, 43, 44, 50, 29, 45, 51, 52, 21, 22, 14]
        allow_grid_list[1] = [9, 10, 18, 27, 35, 43, 51, 28, 36, 19, 59, 20, 44, 52]
        allow_grid_list[2] = [43, 44, 52, 53, 61, 62, 45, 36, 9, 10, 18, 19, 27, 28, 29, 30,\
                             31, 39, 47, 55, 34, 35, 42, 33, 50, 58, 59, 60, 51, 41, 37, 38, \
                             40, 54, 63, 46, 32, 21, 22, 14, 6, 26, 25, 15, 20, 23, 17, 24, 13]
        allow_grid_list[3] = [32, 33, 41, 48, 56, 42, 44, 45, 50, 51, 52, 46, 23, 31, 36, 37,\
                              38, 39, 22, 30, 28, 27]
        allow_grid_list[4] = [34, 35, 36, 44, 52, 53, 61, 14, 15, 22, 23, 28, 29, 30, 37, 38, \
                              27, 26, 24, 32, 40, 31, 39, 45, 46, 25, 33, 41, 60, 47, 48, 49, \
                              56, 5, 13, 20, 21, 16]

    elif(args.scene_lstm_8):
        allow_grid_list[0] = [40, 18, 22, 21, 31, 24, 25, 17]
        allow_grid_list[1] = [63, 13, 21, 11, 10, 20, 26, 27]
        allow_grid_list[2] = [26, 17, 20, 59, 2, 19, 30, 57]
        allow_grid_list[3] = [44, 36, 34, 26, 56, 52, 25, 24]
        allow_grid_list[4] = [28, 16, 19, 34, 48, 35, 17, 43]

    elif(args.scene_lstm_16):
        allow_grid_list[0] = [55, 19, 29, 13, 41, 30, 20, 28, \
                              40, 18, 22, 21, 31, 24, 25, 17]
        allow_grid_list[1] = [53, 54, 55, 56, 57, 58, 61, 62, \
                              63, 13, 21, 11, 10, 20, 26, 27]
        allow_grid_list[2] = [14, 46, 1, 25, 10, 33, 29, 27,  \
                              26, 17, 20, 59, 2, 19, 30, 57]
        allow_grid_list[3] = [45, 37, 53, 33, 16, 28, 43, 35, \
                              44, 36, 34, 26, 56, 52, 25, 24]
        allow_grid_list[4] = [37, 41, 36, 26, 27, 40, 25, 24, \
                              28, 16, 19, 34, 48, 35, 17, 43]

    elif(args.scene_lstm_32):
        allow_grid_list[0] = [34, 46, 44, 9, 26, 32, 53, 43,  \
                              42, 45, 35, 27, 52, 60, 37, 54, \
                              55, 19, 29, 13, 41, 30, 20, 28, \
                              40, 18, 22, 21, 31, 24, 25, 17]

        allow_grid_list[1] = [53, 54, 55, 56, 57, 58, 61, 62, \
                              63, 13, 21, 11, 10, 20, 26, 27, \
                              52, 28, 44, 35, 43, 18, 19, 51, \
                              36, 9, 60, 3, 4, 12, 59, 45]
        allow_grid_list[2] = [3, 42, 22, 16, 21, 61, 35, 0,  \
                              60, 9, 6, 34, 18, 36, 28, 38,  \
                              14, 46, 1, 25, 10, 33, 29, 27, \
                              26, 17, 20, 59, 2, 19, 30, 57]
        allow_grid_list[3] = [55, 38, 47, 30, 50, 61, 32, 41, \
                              31, 46, 42, 49, 54, 27, 48, 51, \
                              45, 37, 53, 33, 16, 28, 43, 35, \
                              44, 36, 34, 26, 56, 52, 25, 24]
        allow_grid_list[4] = [30, 18, 33, 39, 56, 42, 46, 53, \
                              47, 32, 38, 29, 45, 44, 20, 61, \
                              37, 41, 36, 26, 27, 40, 25, 24, \
                              28, 16, 19, 34, 48, 35, 17, 43]

    elif(args.scene_lstm_64): # all_grids
        for d in range(args.num_total_datasets):
            allow_grid_list[d] = np.arange(0,args.scene_grid_num**2).tolist()

    elif(args.scene_lstm_n8):
        allow_grid_list[0] = [21, 22]
        allow_grid_list[1] = [10, 27, 20]
        allow_grid_list[2] = [17, 19, 20, 26, 59, 30]
        allow_grid_list[3] = [56, 44, 36, 52]
        allow_grid_list[4] = [34, 35, 16, 48, 28]

    elif(args.scene_lstm_n16):
        allow_grid_list[0] = [21, 29, 22]
        allow_grid_list[1] = [10, 27, 20]
        allow_grid_list[2] = [33, 59, 10, 46, 14, 17, 19, 20, 25, 26, 27, 29, 30]
        allow_grid_list[3] = [33, 36, 37, 44, 45, 52, 56, 28]
        allow_grid_list[4] = [34, 35, 36, 37, 40, 41, 16, 48, 24, 25, 26, 27, 28]

    elif(args.scene_lstm_n32):
        allow_grid_list[0] = [37, 42, 43, 44, 45, 52, 21, 22, 29]
        allow_grid_list[1] = [35, 36, 9, 10, 43, 44, 59, 18, 19, 20, 52, 51, 27, 28]
        allow_grid_list[2] = [6, 9, 10, 14, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 38, 42, 46, 59, 60, 61]
        allow_grid_list[3] = [32, 33, 36, 37, 38, 41, 42, 44, 45, 46, 48, 50, 51, 52, 56, 27, 28, 30, 31]
        allow_grid_list[4] = [34, 35, 36, 37, 40, 41, 16, 48, 24, 25, 26, 27, 28]

    elif(args.scene_lstm_nU16): 
        allow_grid_list[0] = [13, 14, 17, 18, 19, 20, 21, 22, 24, 25, 28, 29, 30, 31, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 55]
        allow_grid_list[1] = [9, 10, 11, 13, 18, 19, 20, 21, 26, 27, 28, 35, 36, 43, 44, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63]
        allow_grid_list[2] = [1, 2, 6, 9, 10, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63]
        allow_grid_list[3] = [16, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 48, 50, 51, 52, 53, 56]
        allow_grid_list[4] = [5, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 52, 53, 56, 60, 61]

    else:
        print("error define type of scene lstm")
        exit(1)


        ''' UNCOMMENT THIS FOR REAL CALCULATION, THAT PRODUCES ABOUT RESULTS
        # Find all non-linear trajectories and which dataset these non-linear
        # trajectories belongs to 
        for i in range(0,data_loader.num_train_batches):
            # Load batch training data 
            batch  = data_loader.next_batch(randomUpdate=False)
            # Extract non-linear trajectories 
            nonlinearTraj = get_nonlinear_trajectories(batch,args)
            
            #Save nonlinear trajectories
            fig = plt.clf()
            for j in range(len(nonlinearTraj)):
                plt.plot(nonlinearTraj[0][:,0], nonlinearTraj[0][:,1],'ro-')
                plt.axis([-1, 1, -1, 1])
                plt.savefig("./non_linear_trajectories/v{}_b{}_tr{}.png".format(batch["dataset_id"],i,j))
                plt.close()

            # Find which grids, the non linear trajectories pass by 
            if(len(nonlinearTraj) > 0 ):
                list_of_grid_id = find_nonlinear_grids(nonlinearTraj, batch ,args)
                for grid in list_of_grid_id:
                    if (grid not in allow_grid_list[batch["dataset_id"]]):
                        allow_grid_list[batch["dataset_id"]].append(grid)
            
            '''
    return allow_grid_list

def find_nonlinear_grids(nonlinearTraj, batch ,args):

    width_low, width_high = -1 , 1          #scene is in range [-1,1]
    height_low, height_high = -1 , 1        #scene is in range [-1,1]
    boundary_size = 2 
  
    list_of_grid_id = []

    # Process each trajectories    
    for tr in range(len(nonlinearTraj)):

        thisTraj = nonlinearTraj[tr]
        #  Find which grid each location belongs
        for loc in range(thisTraj.shape[0]):

            current_x, current_y = thisTraj[loc,0], thisTraj[loc,1]
            # calculate the grid cell
            cell_x = int(np.floor(((current_x - width_low)/boundary_size) * args.scene_grid_num))
            cell_y = int(np.floor(((current_y - height_low)/boundary_size) * args.scene_grid_num))

            # Peds locations must be in range of [-1,1], so the cell used must be in range [0,scene_grid_num-1]
            if(cell_x < 0):
                cell_x = 0
            if(cell_x >= args.scene_grid_num):
                cell_x = args.scene_grid_num - 1
            if(cell_y < 0):
                cell_y = 0 
            if(cell_y >= args.scene_grid_num):
                cell_y = args.scene_grid_num - 1

            list_of_grid_id.append(cell_x + cell_y*args.scene_grid_num)

    return np.unique(list_of_grid_id)




