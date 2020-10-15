# Description: Function to measure grid diversity - the levels of chaotic motion
# in each grid-cell of each video sequences. The outputs is list of grid-cell in order
# of diversity levels. 
# Author: Manh Huynh
# Date: 12/24/2018 

import math
import torch
import argparse
import time
import tqdm
import shutil
import sys
import matplotlib.pyplot as plt

import operator
from torch import nn, optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
from utils.data_loader import *
from config import *



def measure_grid_diversity(): 

    number_grid_cells = 8 ;         # 8x8 grid_cells
    number_sub_grids = 8 ;          # 8x8 number inner grid_ceslls in each grid 

    # Load data 
    logger = Logger(args, train = False) # make logging utility
    data_loader = DataLoader(args, logger, train = False)

    # a grid-cell is further divided into sub-grids, 
    # each sub-grid has 8 directions to nearyby grids. 
    list_grid_cells = [[] for i in range(number_grid_cells**2)]
  
    for i in range(0, data_loader.num_test_batches):

        batch  = data_loader.next_test_batch(randomUpdate=False, jump = 20)
        numberFrames = len(batch["frame_list"])

        #print("frame_list =", batch["frame_list"])
        for ped in batch["ped_ids"]: 
            for t in range(0,numberFrames-1):


                # find location of this ped in frame t and t+1
                idxInBatch_t = np.where(batch["ped_ids_frame"][t] == ped)[0]
                idxInBatch_next = np.where(batch["ped_ids_frame"][t+1] == ped)[0]

                # if this ped does not have location in this or next frame,
                # skip to next targets.
                if(idxInBatch_t.size is 0 or idxInBatch_next.size is 0):
                    break   # go to next pedestrian

                loc_t = batch["batch_data_absolute"][t][idxInBatch_t][0]
                loc_next = batch["batch_data_absolute"][t+1][idxInBatch_next][0]

                # find grid-cell 
                grid_cell_indx_t = get_grid_cell_index(loc_t, number_grid_cells, range = [-1,1,-1,1])
                grid_cell_indx_next = get_grid_cell_index(loc_next, number_grid_cells, range = [-1,1,-1,1])

                # if 2 locations are not the same grid-cell then go
                # to next route check
                if(grid_cell_indx_t is not grid_cell_indx_next):
                    continue 
                sub_grid_indx_t = get_sub_grid_cell_index(loc_t, number_grid_cells, number_sub_grids, range =[-1,1,-1,1])
                sub_grid_indx_next = get_sub_grid_cell_index(loc_next, number_grid_cells, number_sub_grids, range =[-1,1,-1,1])

                if not list_grid_cells[grid_cell_indx_t]:
                    # if the grid_cell info is empty, then create new route 
                    # and add to this grid-cell info
                    route = {
                        "subgrid_t":sub_grid_indx_t,               
                        "subgrid_next":  sub_grid_indx_next,
                        "count": 1  
                    }
                    list_grid_cells[grid_cell_indx_t].append(route)
                else: 
                    # else, check if the route is exists. 
                    # if the route exists, then increase counts by one.
                    # if not, create a new route and add to the grid-cell info.
                    check_flag = False 
                    for route in list_grid_cells[grid_cell_indx_t]:
                        if(route["subgrid_t"] == sub_grid_indx_t and route["subgrid_next"] == sub_grid_indx_next):
                           # found the same route exists
                           route["count"] += 1
                           check_flag = True 

                    if check_flag is False:
                        # no same route found, create a new one.
                        route = {
                            "subgrid_t":sub_grid_indx_t,               
                            "subgrid_next":  sub_grid_indx_next,
                            "count": 1  
                        }
                        list_grid_cells[grid_cell_indx_t].append(route)

        #print(list_grid_cells)
        #input("here")
    # Calculate common movements degree in each grid-cell
    # common_degree = sum(counts)/sum(routes)
    list_common_degree = []
    for gi in range(number_grid_cells**2):
        
        # number of routes in a grid-cell
        number_routes = len(list_grid_cells[gi])

        # number of counts in a grid-cell
        number_counts = 0 ;
        for route in list_grid_cells[gi]:
            number_counts = number_counts + route["count"]

        common_info = {
            "grid_cell_id": gi ,
            "number_routes": number_routes,
            "number_counts": number_counts,
            "degree": 0 if number_routes is 0 else number_counts/number_routes   
        }

        list_common_degree.append(common_info)

    for i in list_common_degree:
        print(i) 

    # sort by degree
    sorted_list_by_degree = list_common_degree.copy()
    sorted_list_by_degree.sort(key=operator.itemgetter('degree'))
    # Get degree list for plotting
    degree_list = [list_common_degree[gi]["degree"] for gi in range(len(list_common_degree))]

    sorted_degree_list = [sorted_list_by_degree[gi]["degree"] for gi in range(len(sorted_list_by_degree))]
    sorted_grid_cells = [sorted_list_by_degree[gi]["grid_cell_id"] for gi in range(len(sorted_list_by_degree))]

    #plot_common_movement_degree(degree_list)

    print("most common movments grid-cell index: ")
    print(sorted_grid_cells)
    input("here")


def plot_common_movement_degree(degree_list):

    print(degree_list)
    plt.plot(degree_list)
    plt.xlabel('grid-cell index')
    plt.ylabel('common movement score')
    plt.title('UCY_Zara02')

    plt.show()

def plot_results():
    
    # plot common movements degree (sorted)
    sorted_ETH_Hotel = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19.571428571428573, 19.82051282051282, 19.85185185185185, 20.0, 20.0, 20.0, 20.0, 20.441176470588236, 20.92156862745098, 21.023529411764706, 21.045454545454547, 21.384615384615383, 21.44, 21.5, 21.565217391304348, 21.760416666666668, 21.833333333333332, 21.864197530864196, 22.04, 22.055555555555557, 22.6, 23.07843137254902, 23.39622641509434, 24.022727272727273, 24.982142857142858, 25.333333333333332, 26.338461538461537, 29.6, 37.0, 38.0, 41.95454545454545, 49.945945945945944, 57.93333333333333, 63.214285714285715, 72.76470588235294, 84.03125, 100.0, 114.8, 139.53333333333333, 140.0, 148.03333333333333, 153.33333333333334, 196.71428571428572])
    sorted_ETH_Univ = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19.647058823529413, 19.8, 20.0, 20.0, 20.0, 20.65, 21.01123595505618, 21.472972972972972, 22.865671641791046, 23.0, 24.0, 24.12121212121212, 24.96825396825397, 25.016129032258064, 25.106060606060606, 25.232142857142858, 26.6, 26.760416666666668, 28.571428571428573, 30.0, 48.94736842105263, 60.0, 540.0])
    sorted_UCY_Univ = np.array([0, 0, 0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.54054054054054, 21.23076923076923, 21.568627450980394, 21.818181818181817, 22.22222222222222, 22.53968253968254, 22.580645161290324, 22.830188679245282, 22.962962962962962, 23.333333333333332, 23.571428571428573, 23.582089552238806, 23.68421052631579, 24.04494382022472, 24.210526315789473, 24.25531914893617, 24.34108527131783, 25.604395604395606, 25.693430656934307, 25.88235294117647, 25.89041095890411, 26.19047619047619, 26.717557251908396, 26.98181818181818, 27.005649717514125, 27.441860465116278, 27.632850241545892, 27.636363636363637, 27.741935483870968, 28.37837837837838, 28.405797101449274, 28.650793650793652, 29.677419354838708, 30.0, 30.416666666666668, 30.573248407643312, 32.17391304347826, 32.441471571906355, 32.7038626609442, 33.10344827586207, 33.142857142857146, 33.27433628318584, 34.18181818181818, 34.335260115606935, 35.55555555555556, 35.632183908045974, 36.16161616161616, 37.27699530516432, 38.32512315270936, 39.914529914529915, 40.78125, 41.0, 43.63636363636363, 43.956043956043956, 48.927038626609445, 77.5])
    sorted_UCY_Zara01 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19.5, 19.875, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.869565217391305, 20.918367346938776, 21.217391304347824, 21.73134328358209, 22.5, 22.641025641025642, 23.063063063063062, 23.352941176470587, 23.45, 23.935185185185187, 23.975, 24.270491803278688, 24.396825396825395, 24.486486486486488, 24.5, 24.591666666666665, 24.59375, 24.840579710144926, 25.136752136752136, 25.44736842105263, 26.03488372093023, 26.362903225806452, 26.42528735632184, 26.434782608695652, 27.181818181818183, 28.236363636363638, 28.605263157894736, 32.38636363636363, 38.98863636363637])
    sorted_UCY_Zara02 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19.0, 19.5, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.428571428571427, 20.5, 21.285714285714285, 21.946428571428573, 23.03448275862069, 24.0, 24.071428571428573, 24.743589743589745, 25.34065934065934, 25.359154929577464, 25.52, 26.26086956521739, 26.30701754385965, 26.419354838709676, 26.539823008849556, 27.410112359550563, 27.505102040816325, 28.3875, 32.935064935064936, 34.675, 35.00990099009901, 38.943661971830984, 42.38461538461539, 43.67088607594937, 51.3134328358209, 53.08045977011494, 76.0, 93.68817204301075, 113.16666666666667, 230.04504504504504]
    )
    
    plt.plot(sorted_ETH_Hotel, label='ETH_Hotel')
    plt.plot(sorted_ETH_Univ, label='ETH_Univ')
    plt.plot(sorted_UCY_Univ, label='UCY_Univ')
    plt.plot(sorted_UCY_Zara01, label='UCY_Zara01')
    plt.plot(sorted_UCY_Zara02, label='UCY_Zara02')
    plt.ylabel('commom movement score')
    plt.xlabel('grid-cell')
    plt.title('Sorted common movement scores')

    plt.axis([0, 64, 0, 200])
    plt.legend()

    plt.show()


def get_grid_cell_index(inputLocation, number_grid_cells, range = [-1,1,-1,1]):


    # Get x and y of the current ped
    x, y = inputLocation[0], inputLocation[1]

    width_min, width_max= range[0] , range[1]         #scene is in range [-1,1]
    height_min, height_max = range[2] , range[3]         #scene is in range [-1,1]

    boundary_size_x = width_max - width_min 
    boundary_size_y = height_max - height_min 

    # calculate the grid cell
    cell_x = int(np.floor(((x - width_min)/boundary_size_x) * number_grid_cells))
    cell_y = int(np.floor(((y - height_min)/boundary_size_y) * number_grid_cells))

    # Peds locations must be in range of [-1,1], so the cell used must be in range [0,scene_grid_num-1]
    if(cell_x >= 0 and cell_x < number_grid_cells and cell_y >= 0 and cell_y < number_grid_cells ):
        return cell_x + cell_y*number_grid_cells


def get_sub_grid_cell_index(inputLocation, number_grid_cells, number_sub_grids, range =[-1,1,-1,1]):


    width_min, width_max = range[0] , range[1]         #scene is in range [-1,1]
    height_min, height_max = range[2], range[3]

    boundary_size_x = width_max - width_min 
    boundary_size_y = height_max - height_min 

    # Find starting index of grid_cell this location belongs to
    grid_cell_indx = get_grid_cell_index(inputLocation, number_grid_cells, range) 
    cell_x_coord, cell_y_coord = get_coordinate_of_grid_cell(grid_cell_indx, number_grid_cells, range)

    grid_range_min_x = 0 
    grid_range_max_x = boundary_size_x/number_grid_cells
    grid_range_min_y = 0 
    grid_range_max_y = boundary_size_y/number_grid_cells

    # Calculate relative location of inputLocation to starting location of grid_cell
    relativeLocation = inputLocation.copy()
    relativeLocation[0] = inputLocation[0] - cell_x_coord 
    relativeLocation[1] = inputLocation[1] - cell_y_coord  

    # Find grid_cell of relative location is sub-grid of inputLocation
    sub_grid_indx = get_grid_cell_index(relativeLocation, number_grid_cells, \
        range = [grid_range_min_x,grid_range_max_x,grid_range_min_y,grid_range_max_y]) 


    return sub_grid_indx


def get_coordinate_of_grid_cell(grid_cell_indx, number_grid_cells, range = [-1,1,-1,1]):
    
    width_min, width_max = range[0] , range[1]         #scene is in range [-1,1]
    height_min, height_max = range[2], range[3]

    boundary_size_x = width_max - width_min 
    boundary_size_y = height_max - height_min 

    cell_x_coord = width_min + (grid_cell_indx % number_grid_cells)*(boundary_size_x/number_grid_cells)
    cell_y_coord = height_min + np.floor(grid_cell_indx / number_grid_cells)*(boundary_size_y/number_grid_cells)

    return cell_x_coord, cell_y_coord


# Test function
def test_get_grid_cell_index():

    inputLocation = np.array([-1,-1])
    grid_cell_indx = get_grid_cell_index(inputLocation,8, range = [-1,1,-1,1])
    print("inputLocation : {}, is at grid_cell_indx {}, (0 is correct)".format(inputLocation, grid_cell_indx))

    inputLocation = np.array([0.1,0.1])
    grid_cell_indx = get_grid_cell_index(inputLocation,8, range = [-1,1,-1,1])
    print("inputLocation : {}, is at grid_cell_indx {}, (36 is correct)".format(inputLocation, grid_cell_indx))

    inputLocation = np.array([0.6,-0.6])
    grid_cell_indx = get_grid_cell_index(inputLocation,8, range = [-1,1,-1,1])
    print("inputLocation : {}, is at grid_cell_indx {}, (14 is correct)".format(inputLocation, grid_cell_indx))


def test_get_coordinate_of_grid_cell():

    grid_cell_indx = 1  
    cell_x_coord, cell_y_coord = get_coordinate_of_grid_cell(grid_cell_indx,8, range = [-1,1,-1,1])
    print("coordinate of grid_cell {} is {},{} (-0.75,-1) is correct". format(grid_cell_indx, cell_x_coord, cell_y_coord))

    grid_cell_indx = 42  
    cell_x_coord, cell_y_coord = get_coordinate_of_grid_cell(grid_cell_indx,8, range = [-1,1,-1,1])
    print("coordinate of grid_cell {} is {},{} (-0.5, 0.25) is correct". format(grid_cell_indx, cell_x_coord, cell_y_coord))

def test_get_sub_grid_cell_index(): 

    inputLocation = np.array([-1,-1])
    grid_cell_indx = get_grid_cell_index(inputLocation,8)
    sub_grid_indx  = get_sub_grid_cell_index(inputLocation, 8, 8)
    print("inputLocation : {}, is at grid_cell_indx {}, sub_grid {}, (grid_cell 0, sub-grid 0 \
        is correct)".format(inputLocation, grid_cell_indx, sub_grid_indx))

    inputLocation = np.array([-0.9,-0.9])
    grid_cell_indx = get_grid_cell_index(inputLocation,8)
    sub_grid_indx  = get_sub_grid_cell_index(inputLocation, 8, 8)
    print("inputLocation : {}, is at grid_cell_indx {}, sub_grid {}, (grid_cell 0, sub-grid 27 \
        is correct)".format(inputLocation, grid_cell_indx, sub_grid_indx))

    inputLocation = np.array([0.1,-0.6])
    grid_cell_indx = get_grid_cell_index(inputLocation,8)
    sub_grid_indx  = get_sub_grid_cell_index(inputLocation, 8, 8)
    print("inputLocation : {}, is at grid_cell_indx {}, sub_grid {}, (grid_cell 12, sub-grid 35 \
        is correct)".format(inputLocation, grid_cell_indx, sub_grid_indx))
if __name__ == '__main__':
    #test_get_sub_grid_cell_index()
    #test_get_coordinate_of_grid_cell()
    #test_get_grid_cell_index()
    measure_grid_diversity()
    #plot_results()
