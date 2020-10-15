import sys
import math
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from skimage.transform import resize
import matplotlib.patches as patches
from utils.metric_conversions import *


# This function plot contour of one 2d gaussian 
def plot_contour(mu,Sigma):
    # Our 2-dimensional distribution will be over variables X and Y
    N = 100
    X = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    F = multivariate_normal(mu, Sigma)
    Z = F.pdf(pos)
    # Create projected  contour plot 
    #fig = plt.figure()
    
    return X, Y , Z



# test plot_contour
#mu = np.array([-0.1, 0.5])
#sigma_x = 0.3
#sigma_y = 0.2
#corr = 0
#Sigma = np.array([[sigma_x**2 , corr], [corr,  sigma_y**2]])
#plot_contour(mu,Sigma)


def plot_trajectory_pts(x_true, x_predict,batch_size, batch_id, save_trajectory_pts_dir ):
   

    fig = plt.figure(figsize=[4,4])
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis([-1, 1, -1, 1])

    for tid in range(0,batch_size):
        plt.plot(x_true[:,tid,0].cpu().data.numpy(),x_true[:,tid,1].cpu().data.numpy(), 'ro-')
        plt.plot(x_predict[:,tid,0].cpu().data.numpy(),x_predict[:,tid,1].cpu().data.numpy(),'g^-')
    #plt.show()
    plt.savefig(save_trajectory_pts_dir  + '/{}.jpg'.format(batch_id))
    plt.close()  


def plot_trajectory_pts_on_images(batch, batch_id, result_pts, predicted_pids, args):

    if(batch["dataset_id"] == 0): 
        img_dir ,width, height = './imgs/eth_hotel/', 720, 576
    elif(batch["dataset_id"] == 1):
        img_dir ,width, height = './imgs/eth_univ/', 640, 480                                                            
    elif(batch["dataset_id"] == 2):
        img_dir ,width, height = './imgs/ucy_univ/', 720, 576
    elif(batch["dataset_id"] == 3):
        img_dir ,width, height = './imgs/ucy_zara01/', 720, 576
    elif(batch["dataset_id"] == 4):
        img_dir ,width, height = './imgs/ucy_zara02/', 720, 576
    else: 
        print("Invalid dataset id")
        sys.exit(0) 

    # Plot the first frame in tsteps    
    #img_name = '{0:06d}.png'.format(int(batch["frame_list"][args.observe_length]))
    #img = plt.imread(img_dir + img_name)
    #implot = plt.imshow(img)

    # Get peds that has obverve length = args.obsever_length 
    # and predict length = args.predict_length 
    selectedPeds = [] 
    for pedId in predicted_pids:
        isSelected = True 
        for t in range(0,args.observe_length + 1):
            presentIdx = np.where( batch["ped_ids_frame"][t] == pedId)[0]
            if(presentIdx.size == 0): # if predicted peds does not have enough
                isSelected = False    # T_obs + T_predict frames
                break 
        if(isSelected):
            selectedPeds.append(pedId)
    selectedPeds = np.asarray(selectedPeds) 


    result_data = []
    for t in range(0,args.observe_length + args.predict_length):

        # Get predicted location and true location of all peds
        x_predict = np.copy(result_pts[t])
        x_true = np.copy(batch["batch_data_absolute"][t]) 

        # Convert to pixels
        x_true[:,0] = np.rint(width*(x_true[:,0] + 1)/2) 
        x_true[:,1] = np.rint(height*(x_true[:,1] + 1)/2) 
        x_predict[:,0] = np.rint(width*(x_predict[:,0] + 1)/2) 
        x_predict[:,1] = np.rint(height*(x_predict[:,1]  + 1)/2) 

        for i, pedId in enumerate(selectedPeds): 
            
            # Find idx of this selected ped in batch and in results 
            idxInBatch = np.where( batch["ped_ids_frame"][t] == pedId)[0]
            
            if( t < args.observe_length):
                idxInPredict = idxInBatch
            else:
                idxInPredict = np.where(predicted_pids == pedId)[0]
            
            if(idxInBatch.size == 0):
                continue 

            # Store ped data
            pedInfo = np.zeros(6) # frame number , ped id , x_predict, y_predict, x_true , y_true
            pedInfo[0], pedInfo[1], pedInfo[2], pedInfo[3] , pedInfo[4], pedInfo[5] =  \
                    batch["frame_list"][t], pedId, x_predict[idxInPredict,0], x_predict[idxInPredict,1], \
                    x_true[idxInBatch,0], x_true[idxInBatch,1]


            result_data.append(pedInfo)


        # plot trajetories ? 
        #plt.plot(x_true[:,0],x_true[:,1], 'ro')
        #plt.plot(x_predict[:,0],x_predict[:,1],'go')
        #plt.show()
    #plt.savefig(args.save_test_result_pts_dir  + '/{}.jpg'.format(batch_id))
    #plt.close()  

    # Write result to file

    if(len(result_data) != 0):
        result_data = np.vstack(result_data[:])
        filename = args.save_test_result_pts_dir  + '/batch_{:06d}.txt'.format(batch_id)
        np.set_printoptions(suppress=True)
        np.savetxt(filename, result_data, delimiter=',', fmt='%.0f') 


# This function plot predicted 2d Gaussian at each predicted steps of all targets
def plot_trajectory_gaussian(x_true, x_predict, res_gaussian, batch_size, batch_id, save_trajectory_gaussian_dir, args):

    plt.figure(1)
    plt.axis('equal')
    x_true = np.array(x_true)
    x_predict = np.array(x_predict)
    for step in range(0, args.predict_length):

        for tid in range(0,batch_size):
            plt.plot(x_true[tid,:,0],x_true[tid,:,1], 'ro-')
            plt.plot(x_predict[tid,:args.observe_length + step,0],x_predict[tid,:args.observe_length + step,1],'go-')
            plt.axis([-1, 1, -1, 1])
        plt.show()  

    
    '''
    pre_Z = np.zeros(100)
    for step in range(0, args.predict_length):
        plt.clf()
        x_true = np.array(x_true)
        x_predict = np.array(x_predict)
        res_gaussian = np.array(res_gaussian)
        for tid in range(0,batch_size):
            mu = x_predict[tid,step + args.observe_length ,:]
            sigma_x = res_gaussian[tid,step,2]
            sigma_y = res_gaussian[tid,step,3]
            corr = 0
            Sigma = np.array([[1*sigma_x**2 , corr], [corr,  1*sigma_y**2]])

            X, Y , Z = plot_contour(mu, Sigma)
            Z = Z/Z.max()
            print(Z.max())
            Z = Z + pre_Z
            pre_Z = Z
            cax = plt.contourf(X, Y, Z, 20, cmap=cm.jet)
            plt.axis([-1, 1, -1, 1])

            plt.show()
        #pre_Z = pre_Z/pre_Z.max()
        #pre_Z = 0.90*pre_Z
        #cbar = plt.colorbar(cax)
        
        plt.savefig(save_trajectory_gaussian_dir  + '/batch_id={}_steps={}.jpg'.format(batch_id,step + args.observe_length))
        #plt.close() 
        #print("batch_size =" ,  batch_size )
        #input("here")
    plt.close()

    '''

def display_grids_lstm(dataset_id  ,scene_c0_list , args):


    if(dataset_id == 0): 
        img_dir ,width, height = './imgs/eth_hotel/000001.png', 720, 576
    elif(dataset_id == 1):
        img_dir ,width, height = './imgs/eth_univ/000001.png', 640, 480
    elif(dataset_id == 2):
        img_dir ,width, height = './imgs/ucy_univ/000001.png', 720, 576
    elif(dataset_id == 3):
        img_dir ,width, height = './imgs/ucy_zara01/000001.png', 720, 576
    elif(dataset_id == 4):
        img_dir ,width, height = './imgs/ucy_zara02/000001.png', 720, 576
    else: 
        print("Invalid dataset id")
        sys.exit(0) 


    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    img = plt.imread(img_dir)
    image_resized = resize(img, (480,480))

    # Grid lines at these intervals (in pixels)
    # dx and dy can be different
    dx, dy = int(480/args.scene_grid_num) , int(480/args.scene_grid_num)
    # Custom (rgb) grid color
    grid_color = [0,0,0]
    red_color = [255,0,0]
 
    grid_size = int(480/args.scene_grid_num)

    image_resized[:,::dy,:] = grid_color
    image_resized[::dx,:,:] = grid_color
    implot = plt.imshow(image_resized)


    #c0_grid_list ~ [grid_size*grid_size,1,1,rnn_size]
    scene_c0_list  = scene_c0_list[dataset_id]
    scene_c0_list = scene_c0_list.cpu().data.numpy()
    scene_c0_list = np.squeeze(scene_c0_list)

    print("scene_c0_list.shape = ", scene_c0_list.shape)
    print("scene_c0_list.sum = ", scene_c0_list.sum())

    for i in range(args.scene_grid_num**2):
        if(scene_c0_list[i].sum()  != 0 ):
            #this grid has been trained
            grid_x = (i%args.scene_grid_num)*grid_size
            grid_y = int(i/args.scene_grid_num)*grid_size
            print("train grid id = ", i )
            ax.add_patch(
                patches.Rectangle(
                (grid_x, grid_y), # (x,y)
                grid_size,  # widht
                grid_size,  # height
                alpha=0.2,
                facecolor="red",
                fill=True      # remove background
                )
            )

    #plt.show()
    plt.savefig('v4_nonlinear_grids.jpg')


def plot_all_train_trajectory(data_loader, dataset_id, args):
    

    if(dataset_id == 0): 
        img_dir ,width, height = './imgs/eth_hotel/', 720, 576
    elif(dataset_id == 1):
        img_dir ,width, height = './imgs/eth_univ/', 640, 480
    elif(dataset_id == 2):
        img_dir ,width, height = './imgs/ucy_univ/', 720, 576
    elif(dataset_id == 3):
        img_dir ,width, height = './imgs/ucy_zara01/', 720, 576
    elif(dataset_id == 4):
        img_dir ,width, height = './imgs/ucy_zara02/', 720, 576
    else: 
        print("Invalid dataset id")
        sys.exit(0) 

    # Plot background image
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    img_name = '{0:06d}.png'.format(1)
    img = plt.imread(img_dir + img_name)

    image_resized = resize(img, (480,480))

    # Grid lines at these intervals (in pixels)
    # dx and dy can be different
    dx, dy = int(480/args.scene_grid_num) , int(480/args.scene_grid_num)
    # Custom (rgb) grid color
    grid_color = [0,0,0]
    red_color = [255,0,0]
    grid_size = int(480/args.scene_grid_num)

    image_resized[:,::dy,:] = grid_color
    image_resized[::dx,:,:] = grid_color
    implot = plt.imshow(image_resized)

    for batch_id in range(0, data_loader.num_train_batches):
        #batch  = data_loader.next_test_batch(randomUpdate= False)
        batch_size = len(x)
        x = Variable(torch.from_numpy(np.stack(x,axis = 0)).transpose_(0,1)).float().cuda()
        x_true = x[1:]

        # Denomalized location data 
        x_true = x_true.cpu().data.numpy()

        x_true[:,:,0] = np.rint(480*( x_true[:,:,0] + 1)/2) 
        x_true[:,:,1] = np.rint(480*( x_true[:,:,1] + 1)/2) 


        for tid in range(0,batch_size):
            plt.plot(x_true[:,tid,0],x_true[:,tid,1], 'ro-')

    plt.show()




