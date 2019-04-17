# Scene-LSTM


This code/implementation is available for research purposes. If you are using this code/data for your work, please cite the following paper:

> Manh, Huynh and Alaghband, Gita. **Scene-LSTM: A Model for Human Trajectory Prediction**. *arXiv preprint arXiv:1808.04018 (2018).*

> A new version of this paper will be coming soon. 

# This repository contains
 :heavy_check_mark: Processed data (in pixel and meter metrics) for ETH and UCY datasets. This data was also used in SGAN method.
 ```bash
 ├── data 
 │     ├── pixel/*.txt
 │     ├── meter/*.txt
  ```
 :heavy_check_mark: Scripts to convert pixel to meter and vice versa. Double check the paths to 
 homography matrices and input files.

 ```bash
 ├── data_utils
 │     ├── homography_matrix/*.txt
 │     ├── eth_utils/*.m (matlab scripts to process eth data)
 │     ├── data_utils/*.m (matlab scripts to process ucy data))
 ```
- Visualization scripts. (will be available soon) 
- My implementation for LSTM. (will be available soon).
- My implementation for Social-LSTM. (will be available soon)
- My implementation for Scene-LSTM.  (will be available soon)

# Dependencies
The code was tested on Centos 7.0 with python 3.6.8 and pytorch 1.0.0.\
The enviroment was setup using conda 4.5.12. 




