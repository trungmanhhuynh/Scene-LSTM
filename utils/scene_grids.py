import torch
from torch.autograd import Variable

import numpy as np

def distance_point_to_line(A, B, C):

    """
    function to calculate distance from  C to vec AB
    using Pythagoras Theorem d = sqrt(AC^2 - AO^2)
    """ 

    AC =  C - A; 
    AB =  B - A ; 
    magAB = np.sqrt(AB[0]**2 + AB[1]**2 ) 
    if(magAB == 0): return 0            # A and B become one point.
    magAO =  np.dot(AC,AB)/magAB 
    magAC =  np.sqrt(AC[0]**2 + AC[1]**2)
    dist = np.sqrt(abs(magAC**2 - magAO**2))

    return dist 

def check_non_linear_trajectory_v1(traj, thresh = 0.2):

    """
    function: check non-linear trajectories
    input: traj   - list of 2d location (x,y) of a target
            thresh - threshold to check if traj is non-linear
    output: True/False
    """
    non_linear_degree = 0

    y_max = traj[-1,1]
    y_min = traj[0,1]
    y_s   =traj[int(traj.shape[0]/2),1]
    if((y_max-y_min)/2 + y_min == 0):
        non_linear_degree = 0
    else:
        non_linear_degree = abs(((y_max - y_min)/2 + y_min - y_s))

    if(non_linear_degree >= thresh): 
        return True
    else: 
        return False

def check_non_linear_trajectory_v2(traj, thresh = 0.2):

    """
    function: check non-linear trajectories by calculate avarage from
    all middle points to two end points of trajectories.
    input: traj   - list of 2d location (x,y) of a target
        thresh - threshold to check if traj is non-linear
    output: True/False
    """

    numPts = traj.shape[0] -2 # ignore the start and end points
    if(numPts < 3): return False

    startPt = traj[0,:]
    endPt =  traj[-1,:]
    sum_dist = 0 
    list_dist = []
    for pt in traj[1:-1,:]:
        dist = distance_point_to_line(startPt, endPt, pt)
        #sum_dist = sum_dist + dist 
        list_dist.append(dist)

    #avg_dist  = sum_dist/numPts
    #print(avg_dist)
    #if(avg_dist >= thresh): return True 
    if(max(list_dist) >= thresh): return True
    else: return False


def extract_nonlinear_trajectories(batch, args):
    """
    function extracts non-linear trajectories in batch.

    """
    nonlinear_traj = []
    for i, ped in enumerate(batch["all_pids"]): 
        # Gete trajectory of each pedestrian
        traj = [] 
        for t in range(0,args.observe_length + args.predict_length):
            traj.append(batch["loc_abs"][t][i])
        traj = np.vstack(traj[:])
        if(check_non_linear_trajectory_v2(traj,thresh = 0.13)): 
            # Check if this traj is non-linear 
            nonlinear_traj.append(traj)

    return nonlinear_traj


def get_grid_cell_index(loc, num_grid_cells, range = [-1,1,-1,1]):

    """
    function to find grid_cell index of loc (x,y)
    """

    w_min, w_max= range[0] , range[1]         #scene is in range [-1,1]
    h_min, h_max = range[2] , range[3]         #scene is in range [-1,1]

    # calculate the grid cell
    cell_x = int(np.floor(((loc[0] - w_min)/(w_max - w_min)) * num_grid_cells))
    cell_y = int(np.floor(((loc[1] - h_min)/(h_max - h_min)) * num_grid_cells))

    # Peds locations must be in range of [-1,1], so the cell used must be in range [01]
    if(cell_x >= 0 and cell_x < num_grid_cells and cell_y >= 0 and cell_y < num_grid_cells ):
        return cell_x + cell_y*num_grid_cells
    else:
        return -1


def get_sub_grid_cell_index(loc, num_grid_cells, num_sub_grids, range =[-1,1,-1,1]):

    """
       function finds sub_grid locations (x,y)
    """

    w_min, w_max = range[0] , range[1]         #scene is in range [-1,1]
    h_min, h_max = range[2], range[3]

    # Find starting index of grid_cell this location belongs to
    grid_cell_indx = get_grid_cell_index(loc, num_grid_cells, range) 
    cell_x_coord, cell_y_coord = get_coordinate_of_grid_cell(grid_cell_indx, num_grid_cells, range)

    grid_range_min_x = 0 
    grid_range_max_x = (w_max - w_min)/num_grid_cells
    grid_range_min_y = 0 
    grid_range_max_y = (h_max - h_min)/num_grid_cells

    # Calculate relative location of loc to starting location of grid_cell
    relativeLocation = loc.copy()
    relativeLocation[0] = loc[0] - cell_x_coord 
    relativeLocation[1] = loc[1] - cell_y_coord  

    # Find grid_cell of relative location is sub-grid of loc
    sub_grid_indx = get_grid_cell_index(relativeLocation, num_sub_grids, \
        range = [grid_range_min_x,grid_range_max_x,grid_range_min_y,grid_range_max_y]) 

    return sub_grid_indx


def get_coordinate_of_grid_cell(grid_cell_indx, num_grid_cells, range = [-1,1,-1,1]):
    
    """
    functions get coordinate (x,y) at top-left of a grid-cell

    """
    w_min, w_max = range[0] , range[1]         #scene is in range [-1,1]
    h_min, h_max = range[2], range[3]

    boundary_size_x = w_max - w_min 
    boundary_size_y = h_max - h_min 

    cell_x_coord = w_min + (grid_cell_indx % num_grid_cells)*(boundary_size_x/num_grid_cells)
    cell_y_coord = h_min + np.floor(grid_cell_indx / num_grid_cells)*(boundary_size_y/num_grid_cells)

    return cell_x_coord, cell_y_coord


def get_nonlinear_grids(data_loader, args):
    """
    function to find non-linear grids in trained datasets. 

    """
    # Intialize nonlinear_grid_list
    nonlinear_grid_list = [[] for i in range(args.max_datasets)]
    nonlinear_sub_grids_maps =  [[[] for j in range(args.num_grid_cells**2)] for i in range(args.max_datasets)]

    # Find all non-linear trajectories and which dataset these non-linear
    # trajectories belongs to 
    for i in range(0,data_loader.num_train_batches):

        # Load batch training data 
        batch  = data_loader.next_batch(randomUpdate=False)

        # Extract non-linear trajectories 
        nonlinearTraj = extract_nonlinear_trajectories(batch,args)

        for tr in nonlinearTraj:
            # Process each trajectories    
            for loc in tr:
                # Find grid_cell index of each location and add it to the list
                gid = get_grid_cell_index(loc, args.num_grid_cells, range = [-1,1,-1,1])
                if(gid == -1): continue 
                if(gid not in nonlinear_grid_list[batch["dataset_id"]]):
                    nonlinear_grid_list[batch["dataset_id"]].append(gid)

                sub_id = get_sub_grid_cell_index(loc, args.num_grid_cells, args.num_sub_grids, range =[-1,1,-1,1])
                if(sub_id ==-1): continue 
                if(sub_id not in nonlinear_sub_grids_maps[batch["dataset_id"]][gid]):
                    nonlinear_sub_grids_maps[batch["dataset_id"]][gid].append(sub_id)

        #Save nonlinear trajectories
        #fig = plt.clf()
        #for j in range(len(nonlinearTraj)):
        #    plt.plot(nonlinearTraj[0][:,0], nonlinearTraj[0][:,1],'ro-')
        #    plt.axis([-1, 1, -1, 1])
        #    plt.savefig("./non_linear_trajectories/v{}_b{}_tr{}.png".format(batch["dataset_id"],i,j))
        #   plt.close()

    return nonlinear_grid_list, nonlinear_sub_grids_maps

def get_route_info(data_loader, args):
    """
    function extract route information in each grid_cell of each train dataset.
    A route is a path from one sub_grid to another sub_grid.

    """
    route_info = [[[] for j in range(args.num_grid_cells**2)] for i in range(args.max_datasets)]

    for i in range(0,data_loader.num_train_batches):

        # Load batch training data
        batch  = data_loader.next_batch(randomUpdate=False, jump = args.predict_length + args.observe_length)
        numberFrames = len(batch["frame_list"])

        for ped in batch["all_pids"]: 
            for t in range(0,numberFrames-1):

                # find location of this ped in frame t and t+1
                idxInBatch_t = np.where(batch["all_pids"] == ped)[0]
                idxInBatch_next = np.where(batch["all_pids"] == ped)[0]

                # if this ped does not have location in this or next frame,
                # continue to next targets.
                if(idxInBatch_t.size is 0 or idxInBatch_next.size is 0): break   # go to next pedestrian

                loc_t = batch["loc_abs"][t][idxInBatch_t][0]
                loc_next = batch["loc_abs"][t+1][idxInBatch_next][0]

                # find grid-cell 
                gi_t = get_grid_cell_index(loc_t, args.num_grid_cells, range = [-1,1,-1,1])
                gi_next = get_grid_cell_index(loc_next, args.num_grid_cells, range = [-1,1,-1,1])
                if(gi_t != gi_next or gi_t == -1 or gi_next == -1): continue 

                # find sub_grid 
                sub_t = get_sub_grid_cell_index(loc_t, args.num_grid_cells, args.num_sub_grids, range =[-1,1,-1,1])
                sub_next = get_sub_grid_cell_index(loc_next, args.num_grid_cells, args.num_sub_grids, range =[-1,1,-1,1])

                # check if the route is exists. 
                # if the route exists, then increase counts by one.
                # if not, create a new route and add to the grid-cell info.
                check_flag = False 
                for route in route_info[batch["dataset_id"]][gi_t]:
                    if(route["subgrid_t"] == sub_t and route["subgrid_next"] == sub_next):
                        # found the same route exists
                        route["count"] += 1
                        check_flag = True 
                        break  

                if check_flag is False:
                    # no same route found, create a new one.
                    route = {
                        "subgrid_t":sub_t,               
                        "subgrid_next":  sub_next,
                        "count": 1  
                    }
                    route_info[batch["dataset_id"]][gi_t].append(route)

    return route_info

def get_common_grids(data_loader, args):

    """
    function to find common movement grids in trained datasets. 

    """
    # Intialize common_grid_list
    common_grid_list = [[] for i in range(args.max_datasets)]
    common_sub_grids_maps =  [[[] for j in range(args.num_grid_cells**2)] for i in range(args.max_datasets)]
    
    # Calculate route information
    route_info = get_route_info(data_loader, args) 

    # Calculate common movements score in each grid-cell
    # mov_score = sum(counts)/sum(routes)
    mov_score =  [[[] for j in range(args.num_grid_cells**2)] for i in range(args.max_datasets)]
    common_info =  [[[] for j in range(args.num_grid_cells**2)] for i in range(args.max_datasets)]

    for dataset_id in range(args.max_datasets):
        for gi in range(args.num_grid_cells**2):
            
            # number of routes in a grid-cell
            number_routes = len(route_info[dataset_id][gi])

            # number of counts in a grid-cell
            number_counts = 0 ;
            for route in route_info[dataset_id][gi]:
                number_counts = number_counts + route["count"]
            
            mov_score[dataset_id][gi] = 0 if number_routes is 0 else number_counts/number_routes

    # Get common_grid_list
    for dataset_id in range(args.max_datasets):

        sorted_grids = [i[0] for i in sorted(enumerate(mov_score[dataset_id]), key=lambda x:x[1], reverse=True)]
        if(dataset_id in args.train_dataset):
            common_grid_list[dataset_id] = sorted_grids[0:args.num_common_grids]
            for gi in common_grid_list[dataset_id]:
                for route in route_info[dataset_id][gi]:
                    if(route["count"] >= 3 and route["subgrid_t"] not in common_sub_grids_maps[dataset_id][gi]):
                        common_sub_grids_maps[dataset_id][gi].append(route["subgrid_t"])
                    if(route["count"] >= 3 and route["subgrid_next"] not in common_sub_grids_maps[dataset_id][gi]):
                        common_sub_grids_maps[dataset_id][gi].append(route["subgrid_next"])


    return common_grid_list, common_sub_grids_maps
