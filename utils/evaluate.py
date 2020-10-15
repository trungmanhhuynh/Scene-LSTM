import random
import math
import torch
import argparse
import time
import tqdm
import shutil

from torch import nn, optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from utils.metric_conversions import *
from utils.scene_grids import check_non_linear_trajectory_v2


# Function to calculate MSE for each batch
def calculate_mean_square_error(batch, results, predicted_pids, args):

	# Process each predicted frame
	mseTotal = 0 	
	for t in range(args.observe_length, args.observe_length + args.predict_length):

		# Process for each ped 
		mseFrame = 0 

		# Get predicted location and true location
		predict_loc = results[t]
		true_loc = batch["loc_abs"][t]

		# convert metrics
		predict_loc = convert_input2output_metric(predict_loc, batch["dataset_id"], args.input_metric, args.output_metric)
		true_loc = convert_input2output_metric(true_loc, batch["dataset_id"], args.input_metric, args.output_metric)
	
		# Calculate mse in each frame of a ped
		msePed = np.sqrt((predict_loc[:,0]- true_loc[:,0])**2 +  (predict_loc[:,1]- true_loc[:,1])**2)	

		# Calculate mse of all peds in a frame 
		mseTotal = mseTotal + msePed.sum()

	return mseTotal/(args.predict_length*len(predicted_pids))

# Function to calculate MSE for each non-linear batch
def calculate_mean_square_error_nonlinear(batch, results, predicted_pids, args):

   # Find Peds with non-linear trajectories 
	nonLinearPeds = []
	for i, p in enumerate(predicted_pids):
		# Get trajectory of this target
		traj = [batch["loc_abs"][ti][i] for ti in range(0,args.observe_length + args.predict_length)]
		
		# check if ped's trajectory is non-linear 
		if(check_non_linear_trajectory_v2(np.asarray(traj), thresh = 0.10)): 
			nonLinearPeds.append(p)

	if(len(nonLinearPeds) == 0):				# If there is no non-linear trajectories
		return False, 0
 
	mseTotal = 0 		
	for t in range(args.observe_length, args.observe_length + args.predict_length):
		for pedId in nonLinearPeds:

	    	# Get predicted location and true location
			idx = np.where(batch["all_pids"] == pedId)[0]
			predict_loc = results[t][idx]
			true_loc = batch["loc_abs"][t][idx]
				
			# convert metrics
			predict_loc = convert_input2output_metric(predict_loc, batch["dataset_id"], args.input_metric, args.output_metric)
			true_loc = convert_input2output_metric(true_loc, batch["dataset_id"], args.input_metric, args.output_metric)

			# Calculate mse in each frame of a ped
			msePed = np.sqrt((predict_loc[:,0]- true_loc[:,0])**2 +  (predict_loc[:,1]- true_loc[:,1])**2)	
			mseTotal = mseTotal + msePed[0]

	return True, mseTotal/(len(nonLinearPeds)*args.predict_length)

def calculate_final_displacement_error(batch, results, predicted_pids, args):

	mseTotal = 0 									# Initilize mse  = 0 

	# Get predicted location and true location
	predict_loc = results[args.observe_length + args.predict_length - 1]
	true_loc = batch["loc_abs"][args.observe_length + args.predict_length - 1] 
		
	# convert metrics
	predict_loc = convert_input2output_metric(predict_loc, batch["dataset_id"], args.input_metric, args.output_metric)
	true_loc = convert_input2output_metric(true_loc, batch["dataset_id"], args.input_metric, args.output_metric)

	# Calculate mse in each frame of a ped
	msePed = np.sqrt((predict_loc[:,0]- true_loc[:,0])**2 +  (predict_loc[:,1]- true_loc[:,1])**2)	

	# Calculate mse of all peds in a frame 
	mseTotal = mseTotal + msePed.sum()

	return mseTotal/len(predicted_pids)


