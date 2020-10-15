import numpy as np
from utils.metric_conversions import *

def save_trajectories(bid, batch, results, predicted_pids, args):

    result_dict = [] 
    obs_traj = [] 
    for t in range(0, args.observe_length):
        true_loc = batch["loc_abs"][t]
        true_loc = convert_input2output_metric(true_loc, batch["dataset_id"], 'normalized_pixels', 'pixels')
        obs_traj.append(true_loc.tolist())

    predict_traj = [] 
    gt_traj = [] 
    for t in range(args.observe_length, args.observe_length + args.predict_length):

        predict_loc = results[t]
        true_loc = batch["loc_abs"][t]
        # convert metrics
        predict_loc = convert_input2output_metric(predict_loc, batch["dataset_id"], 'normalized_pixels', 'pixels')
        true_loc = convert_input2output_metric(true_loc, batch["dataset_id"], 'normalized_pixels', 'pixels')
        predict_traj.append(predict_loc.tolist())
        gt_traj.append(true_loc.tolist())

    result_dict = {
        'batch_id': bid,
        'frame_id': batch['frame_list'].tolist(),
        'obs_traj': obs_traj,
        'traj_gt': gt_traj,
        'pred_traj':predict_traj
    }

    return result_dict