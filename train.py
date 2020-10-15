import random
import math
import torch
import argparse
import time
import tqdm
import shutil
import os 

from torch import nn, optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np

from models.Model_LSTM import Model_LSTM

from utils.evaluate import  *
from utils.data_loader import DataLoader, Logger
from utils.visualize import *
from config import get_args
from sample import sample


def run_model(model, data_loader, logger, args):

    #Create model 
    torch.manual_seed(1)
    net = model(data_loader, args, train = True)
    if(args.use_cuda): net = net.cuda() 
    optimizer = optim.RMSprop(net.parameters(), lr = args.learning_rate)
    print(net)
   
    best_mse_val, best_epoch, start_epoch= 10e5, 10e5, 0  
    for e in range(start_epoch, args.nepochs):
        epoch_loss = 0                                                           # Init loss of this epoch
        data_loader.shuffle_data()                                               # Shuffle data
        for i in range(0,data_loader.num_train_batches):
            # forward/backward for each train batch
            start = time.time()                                                  # Start timer
            batch  = data_loader.next_batch(randomUpdate=True)                  # Load batch training data 
            net.init_batch_parameters(batch)                                     # 
            net.init_target_hidden_states()                                      # Init hidden states for each target in this batch
            optimizer.zero_grad()                                                # Zero out gradients
            batch_loss = 0                                                       # Init loss of this batch

            for t in range(args.observe_length + args.predict_length - 1):
            
                # Get batch data at time t and t+1
                cur_frame = {"loc_off": batch["loc_off"][t], "loc_abs": batch["loc_abs"][t], "frame_pids":  batch["all_pids"]}
                next_frame = {"loc_off": batch["loc_off"][t+1],"loc_abs": batch["loc_abs"][t+1], "frame_pids":  batch["all_pids"]}

                # Process current batch and produce loss
                loss_t = net.process(cur_frame, next_frame)
                batch_loss = batch_loss + loss_t

            # back ward
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss = epoch_loss + batch_loss.item()/(args.observe_length + args.predict_length) # Add batch_loss to epoch loss
            stop = time.time()                                                  # Stop timer

            # logging
            if (data_loader.num_train_batches* e + i) % args.info_freq == 0:
                logger.write('iter:{}/{}, batch_loss:{:f}, time/iter:{:.3f} ms, time left: {:.3f} hours' \
                            .format(data_loader.num_train_batches* e + i, data_loader.num_train_batches*args.nepochs,
                            epoch_loss/(i+1),  (stop-start)*1000,
                            (data_loader.num_train_batches*args.nepochs - data_loader.num_train_batches* e + i)*(stop-start)/3600))


        epoch_loss = epoch_loss/data_loader.num_train_batches                   # Loss of this epoch

        # Save model in each epoch
        save_model_file = '{}/net_epoch_{:06d}.pt'.format(args.save_model_dir,e)
        state = {'epoch': e, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state,save_model_file)   
        logger.write("saved model to " +  save_model_file)

    #--------------------------------------------------------------------------------------------$
    #                             VALIDATION SECTION 
    #--------------------------------------------------------------------------------------------$
        logger.write('Calculating MSE on validation data ......')
        mse_eval, nde_eval, fde_eval = sample(model, data_loader, save_model_file, args, validation = True)

        # Copy to best model if having better mse_eval 
        if(mse_eval < best_mse_val): 
            best_mse_val, best_epoch = mse_eval, e 
            best_nde_eval = nde_eval
            best_fde_eval = fde_eval
            best_epoch_model_dir = '{}/best_epoch_model.pt'.format(args.save_model_dir)          
            shutil.copy(save_model_file,best_epoch_model_dir)                  # Save best_model file

        # Print out results
        logger.write('epoch: {}, epoch_loss: {}, mse_eval: {:.3f}'.format(e, epoch_loss , mse_eval), record_loss = True)
        logger.write('best_epoch: {}, mse_eval: {:.3f}, nde_eval: {:.3f}, fde_eval: {:.3f}'\
                    .format(best_epoch, best_mse_val, best_nde_eval, best_fde_eval))
        
def get_model(args):
    model_dict={
        "Model_LSTM": Model_LSTM
    }
    return model_dict[args.model_name]

if __name__ == '__main__':

    args = get_args()          # Get input argurments 
    args.log_dir = os.path.join(args.save_root, args.model_dir, str(args.model_dataset), 'log')
    args.save_model_dir =  os.path.join(args.save_root, args.model_dir, str(args.model_dataset), 'model')
    
    logger = Logger(args, train = True)                          # make logging utility
    logger.write("{}\n".format(args))

    data_loader = DataLoader(args, logger, train = True)         # load data
    model = get_model(args)                                      # get model name
    run_model(model, data_loader, logger, args)                  # train + validate
