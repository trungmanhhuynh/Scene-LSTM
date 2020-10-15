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

from models.Model_LSTM import Model_LSTM

from utils.evaluate import *
from utils.data_loader import *
from utils.visualize import *
from config import *
from sample import *

def get_model(args):
    model_dict={
        "Model_LSTM": Model_LSTM
    }
    if(model_dict[args.model_name] is Model_LSTM or model_dict[args.model_name] is Model_LSTM_abs or 
        model_dict[args.model_name] is Social_LSTM_abs or model_dict[args.model_name] is Model_Linear):
        assert args.use_nonlinear_grids is False
        assert args.use_subgrid_maps is False 
        assert args.use_scene is False
    if(model_dict[args.model_name] is Model_LSTM_Scene or model_dict[args.model_name] is Model_LSTM_Scene_SoftFilter):
        assert args.use_nonlinear_grids is False
        assert args.use_subgrid_maps is False 
        assert args.use_scene is True
    if(model_dict[args.model_name] is Model_LSTM_Scene_nonlinear or model_dict[args.model_name] is Model_LSTM_Scene_nonlinear_noSF):
        assert args.use_nonlinear_grids is True    
        assert args.use_subgrid_maps is False 
        assert args.use_scene is True
    if(model_dict[args.model_name] is Model_LSTM_Scene_nonlinear_subgrids or model_dict[args.model_name] is Model_LSTM_Scene_nonlinear_subgrids_noSF):
        assert args.use_nonlinear_grids is True    
        assert args.use_subgrid_maps is True   
        assert args.use_scene is True

    return model_dict[args.model_name]

if __name__ == '__main__':

    args = get_args()          # Get input argurments/
    args.log_dir = os.path.join(args.save_root , args.dataset_size, args.model_dir, str(args.model_dataset), 'log')
    args.save_model_dir =  os.path.join(args.save_root , args.dataset_size, args.model_dir, str(args.model_dataset), 'model')
    args.save_traj_dir =  os.path.join(args.save_root , args.dataset_size, args.model_dir, str(args.model_dataset), 'test_result_pts.txt')

    logger = Logger(args, train = False)                 # make logging utility
    logger.write("{}\n".format(args))
    data_loader = DataLoader(args, logger, train = False)
    model = get_model(args)

    logger.write('evaluating on test data ......')
    save_model_file = '{}/best_epoch_model.pt'.format(args.save_model_dir)
    mse_eval, nde_eval, fde_eval = sample(model, data_loader, save_model_file, args, test = True)

    # Print out results
    logger.write('mse_eval: {:.3f}, nde_eval: {:.3f}, fde_eval: {:.3f}'.format(mse_eval, nde_eval, fde_eval))