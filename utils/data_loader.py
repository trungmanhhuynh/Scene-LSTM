'''
Author : Huynh Manh
Date : 
'''

import os
import sys
import pickle
import numpy as np
import random
import math
import time
from datetime import datetime
from utils.video_info import get_video_list

class DataLoader():

    def __init__(self, args, logger , train= False):

        self.data_dir = './processed_normalized_pixel_data'                   # Where the pre-processed pickle file resides             
        self.data_dir = os.path.join(self.data_dir, args.dataset_size)      

        # List of data directories where raw data resides
        video_list = get_video_list()
        self.num_datasets = len(video_list) 
        dataset_dirs = [os.path.join(self.data_dir, video['name']) for video in video_list]
        print("Dataset used :", dataset_dirs )

        self.logger = logger           
        #self.used_datasets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        if(train):
            # all the train_dataset will be used to train the model
            # which then will be used for testing the model_dataset
            self.train_dataset = args.train_dataset
            self.valid_dataset = args.train_dataset
            self.val_fraction =  0.2 # 20% batches used for validation  
            self.train_fraction =  0.8 # 80% batches used for training  

            if(args.stage2):
                self.val_fraction =  0.5  # 50% batches used for validation  
                self.train_fraction =  0.5 #50% batches used for training  

        else:
            self.test_dataset = [args.model_dataset]
            self.test_fraction = 1 # 50% batches used for testing  
        
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")          # Where the pre-processed pickle file resides

        # Assign testing flag 
        self.train = train
        self.pre_process = args.pre_process
        self.tsteps = args.observe_length + args.predict_length

        # 
        self.num_batches = 0 ; 
        self.num_train_batches = 0 ; 
        self.num_validation_batches = 0 ; 
        self.num_test_batches = 0 ; 

        # If the file doesn't exist
        if (not(os.path.exists(data_file)) or self.pre_process):
            print("Pre-processing data from raw data")
            self.frame_preprocess(dataset_dirs, data_file)         #Preprocess data file       

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)

        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(train = True, valid = True , test = True)


    def frame_preprocess(self, dataset_dirs, data_file):
        '''
        This function splits video data into batches of frames. Each batch
        is a dictionary. 
        + Each dataset will have its own set of batches, stored in all_data 
        '''
        # Intialize output values 
        all_batches = [[] for i in range(self.num_datasets)]            # batch data of each dataset

        # Proceed each dataset at a time
        for dataset_id, directory in enumerate(dataset_dirs):
        
            # Load the data from the  file
            file_path = os.path.join(directory, 'normalized_data.txt')
            data = np.genfromtxt(file_path, delimiter=',')

            # Read frame-by-frame and store them to a temporary data for this dataset 
            dataset_data = [] 
            frameList = np.unique(data[:,0]).tolist()      
            for frame in frameList:
                pedDataInFrame = data[data[:,0] == frame,:]   # Extract all pedestrians in current frame                                         
                dataset_data.append(pedDataInFrame)           # Store it to temporary dataset_data

            # Gather batch data (by frame) of pedestrians have trajectories = args.predict_length + args.observed_length 
            # Each batch slides by 1 frame 
            for i in range(0, len(dataset_data) - self.tsteps):

                # Initialize batch as a dictionary 
                batch = {"loc_abs":[],"loc_off":[],"all_pids":[],"frame_list":-1,"dataset_id":-1}
                                  
                temp_batch = dataset_data[i: i + self.tsteps]                    # batch of tsteps frames, each ~ [frameId, targetId, x ,y]
                temp_batch = np.vstack(temp_batch)
                sel_pids = self.get_selected_peds(temp_batch)                    # seleted pids have traj = pred + obs length
                if(len(sel_pids) == 0): continue                                 # jump to next frame
                                                                                 # if there is no sastified peds
                batch["frame_list"] = np.unique(temp_batch[:,0])                 # List of frame numbers
                batch["all_pids"]   = np.asarray(sel_pids)                       # List of ped ids
                batch["dataset_id"] = dataset_id                                 # Set dataset id of this batch 
                # Gather absolute locations for each selected pids
                for ind, frameId in enumerate(batch["frame_list"]):
                    frame_data = temp_batch[temp_batch[:,0] == frameId,:]
                    frame_data =  [frame_data[frame_data[:,1] == pid,:] for pid in sel_pids]
                    frame_data = np.vstack(frame_data)
                    # Store absoluate x,y location 
                    batch["loc_abs"].append(frame_data[:,[2,3]])        
            
                batch["loc_off"] = self.calculate_loc_off(batch)            # Get data_offset of this batch
                all_batches[dataset_id].append(batch)                       # Gather into all batches

            # Finish processing for one dataset     
            self.logger.write("{}: #frames: {}, #batches: {}".format(directory, len(frameList), len(all_batches[dataset_id])))

        # Save all_batches in the pickle file
        f = open(data_file, "wb")
        pickle.dump(all_batches, f, protocol = 2)
        f.close()

    def get_selected_peds(self, batch):
        """
            function to find peds that have trajectory length = pred + obs length
        """
        all_pids = np.unique(batch[:,1])            # all ped ids in this batch
        ped_list = []                               # init the selected pid list 
        # Scan through each pid and check if its trajectory sastifies the length
        for pid in all_pids:
            traj_length = np.sum(batch[:,1] == pid)
            if(traj_length == self.tsteps):
                ped_list.append(pid)

        return ped_list

    def calculate_loc_off(self, batch):      
        """
            function calculate offset locations in batch.
        """
        def Cloning(li1):
            li_copy =[]
            for item in li1: li_copy.append(np.copy(item))
            return li_copy
 
        loc_off = Cloning(batch["loc_abs"])                     # Initialize loc_off as loc_abs

        # calculate location offset for each frame.
        # the first frame has location offset = 0
        for t in reversed(range(self.tsteps)):
            if(t == 0): # t is start frame
                loc_off[t][:]  =  0
            else:
                loc_off[t][:] =  loc_off[t][:] -  loc_off[t-1][:]

        return loc_off

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.all_batches = self.raw_data
        self.num_batches_list = [len(self.all_batches[i]) for i in range(self.num_datasets)]
        print(self.num_batches_list)

        # Intilizing list for each mode
        self.train_batch = [] 
        self.validation_batch = []
        self.test_batch = []
        self.num_train_batches_list = [0]*self.num_datasets
        self.num_validation_batches_list = [0]*self.num_datasets  
        self.num_test_batches_list = [0]*self.num_datasets  

        # For each dataset
        for dataset in range(self.num_datasets):
            self.train_batch.append([]) 
            self.validation_batch.append([])
            self.test_batch.append([])
            
            if(self.train == True):

                if(dataset in self.train_dataset):

                    # get the train data for the specified dataset
                    #self.num_train_batches_list[dataset] = int(self.num_batches_list[dataset]*(1 - self.val_fraction))
                    self.num_train_batches_list[dataset] = int(self.num_batches_list[dataset]*self.train_fraction)
                    self.train_batch[dataset] = self.all_batches[dataset][0:self.num_train_batches_list[dataset]]
                    
                    # get the validation data for the specified dataset
                    self.num_validation_batches_list[dataset] = int(self.num_batches_list[dataset]*self.val_fraction)
                    self.validation_batch[dataset]  = self.all_batches[dataset][-self.num_validation_batches_list[dataset]:]
        
            else:
                if(dataset in self.test_dataset):

                    self.num_test_batches_list[dataset] = int(self.num_batches_list[dataset]*self.test_fraction)
                    self.test_batch[dataset]  = self.all_batches[dataset][-self.num_test_batches_list[dataset]:]

            self.logger.write('---')
            self.logger.write('Training data from dataset {} :{}'.format(dataset, len(self.train_batch[dataset])))
            self.logger.write('Validation data from dataset {} :{}'.format(dataset, len(self.validation_batch[dataset]) ))
            self.logger.write('Test data from dataset {} :{}'.format(dataset, len(self.test_batch[dataset]) ))

        self.logger.write('---')
        self.num_train_batches = sum(self.num_train_batches_list)
        self.num_validation_batches= sum(self.num_validation_batches_list)
        self.num_test_batches = sum(self.num_test_batches_list)
        self.logger.write('Total num_train_batches : {}'.format(sum(self.num_train_batches_list)))
        self.logger.write('Total num_validation_batches : {}'.format(sum(self.num_validation_batches_list)))
        self.logger.write('Total num_test_batches : {}'.format(sum(self.num_test_batches_list)))

    def shuffle_data(self):
        #random.seed(4)
        self.shuffled_dataset = self.train_dataset
        random.shuffle(self.shuffled_dataset)
        self.shuffled_batch = [] 
        for i in self.shuffled_dataset:
            batch_idx  = list(range(0,self.num_train_batches_list[i]))
            random.shuffle(batch_idx)
            self.shuffled_batch.append(batch_idx)

        self.shuffled_dataset_id = 0 
        self.shuffled_batch_id = 0 

    def next_batch(self, randomUpdate=True, jump = 1):
        '''
        Function to get the next batch of points
        '''
        # advance the frame pointer to a random point
        if randomUpdate:
            assert self.shuffled_dataset_id < len(self.shuffled_dataset), "Error:shuffled_dataset_id={}".format(self.shuffled_dataset_id)
            dataset_idx = self.shuffled_dataset[self.shuffled_dataset_id]
            batch_idx = self.shuffled_batch[self.shuffled_dataset_id][self.shuffled_batch_id] 
            batch_data = self.train_batch[dataset_idx][batch_idx]

            # increase shuffled_batch_id by 1, if it is over the len of batch_size of a sequence
            # reset the shuffled_batch_id and increase shuffled_dataset_id by 1
            self.shuffled_batch_id = self.shuffled_batch_id + 1
            if(self.shuffled_batch_id  == len(self.shuffled_batch[self.shuffled_dataset_id])):
                self.shuffled_batch_id = 0 
                self.shuffled_dataset_id = self.shuffled_dataset_id + 1
       
        else:    
            # Extract the frame data of the current dataset
            dataset_idx =self.train_dataset[self.train_dataset_pointer]
        
            # Get the frame pointer for the current dataset
            batch_idx = self.train_batch_pointer
        
            # Number of unique peds in this sequence of frames
            batch_data = self.train_batch[dataset_idx][batch_idx]
            
            self.tick_batch_pointer(train=True, jump = jump)


        return batch_data

    def next_valid_batch(self, randomUpdate=False):
        '''
        Function to get the next batch of points
        '''

       # Extract the frame data of the current dataset
        dataset_idx =self.train_dataset[self.valid_dataset_pointer]

        # Get the frame pointer for the current dataset
        batch_idx = self.valid_batch_pointer
    
        # Number of unique peds in this sequence of frames
        batch_data = self.validation_batch[dataset_idx][batch_idx]

        self.tick_batch_pointer(valid=True)

   
        return batch_data

    def next_test_batch(self, randomUpdate=False, jump = 1):
        '''
        Function to get the next batch of points
        '''
        # Extract the frame data of the current dataset
        dataset_idx =self.test_dataset[self.test_dataset_pointer]
        # Get the frame pointer for the current dataset
        batch_idx = self.test_batch_pointer
    
        # Number of unique peds in this sequence of frames
        batch_data = self.test_batch[dataset_idx][batch_idx]

        self.tick_batch_pointer(test=True, jump = jump)

   
        return batch_data

    def tick_batch_pointer(self, train = False, jump = 1, valid = False , test = False):
        '''
        Advance the dataset pointer
        '''
        if train:
            self.train_batch_pointer += jump                  # Increment batch pointer
            dataset_idx =self.train_dataset[self.train_dataset_pointer]

            if self.train_batch_pointer  >= self.num_train_batches_list[dataset_idx] :
                # Go to the next dataset
                self.train_dataset_pointer += 1
                # Set the frame pointer to zero for the current dataset
                self.train_batch_pointer = 0
                # If all datasets are done, then go to the first one again
                if self.train_dataset_pointer >= len(self.train_dataset):
                    self.reset_batch_pointer(train = True)
        if valid:       
            self.valid_batch_pointer += jump                  # Increment batch pointer
            dataset_idx =self.train_dataset[self.valid_dataset_pointer]
            if self.valid_batch_pointer  >= self.num_validation_batches_list[dataset_idx] :
                # Go to the next dataset
                self.valid_dataset_pointer += 1
                # Set the frame pointer to zero for the current dataset
                self.valid_batch_pointer = 0
                # If all datasets are done, then go to the first one again
                if self.valid_dataset_pointer >= len(self.valid_dataset):
                    self.reset_batch_pointer(valid = True)

        if test:       
            self.test_batch_pointer += jump              # Increment batch pointer
            dataset_idx =self.test_dataset[self.test_dataset_pointer]
            if self.test_batch_pointer  >= self.num_test_batches_list[dataset_idx] :
                # Go to the next dataset
                self.test_dataset_pointer += 1
                # Set the frame pointer to zero for the current dataset
                self.test_batch_pointer = 0
                # If all datasets are done, then go to the first one again
                if self.test_dataset_pointer >= len(self.test_dataset):
                    self.reset_batch_pointer(test = True)



    def reset_batch_pointer(self, train = False, valid = False , test = False):
        '''
        Reset all pointers
        '''
        if train:
            # Go to the first frame of the first dataset
            self.train_dataset_pointer = 0
            self.train_batch_pointer = 0

        if valid:
            self.valid_dataset_pointer = 0
            self.valid_batch_pointer = 0

        if test:
            self.test_dataset_pointer = 0
            self.test_batch_pointer = 0

# abstraction for logging
class Logger():
    def __init__(self, args , train = False):
        
        self.train = train
        if(train):
            # open file for record screen 
            self.train_screen_log_file_path = '{}/train_screen_log_{}.txt'.format(args.log_dir,datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 
            self.train_screen_log_file = open(self.train_screen_log_file_path, 'w')
           
            # open file for recording train loss 
            self.train_log_file_path = '{}/train_log_{}.txt'.format(args.log_dir,datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 
            self.train_log_file= open(self.train_log_file_path, 'w')

        else: 
            self.test_screen_log_file_path = '{}/test_screen_log_{}.txt'.format(args.log_dir,datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 
            self.test_screen_log_file = open(self.test_screen_log_file_path, 'w')

    def write(self, s, record_loss = False):

        print (s)

        if(self.train):
            with open(self.train_screen_log_file_path, 'a') as f:
                f.write(s + "\n")
            if(record_loss):
                with open(self.train_log_file_path, 'a') as f:
                    f.write(s + "\n")
        else:
            with open(self.test_screen_log_file_path, 'a') as f:
                f.write(s + "\n")


