"""
Script to test functions in scene_grids
"""

from scene_grids import *
import matplotlib.pyplot as plt



def test_check_non_linear_trajectory_v1():

    # Assume a non-linear trajectory.
    traj1 = np.array([[0.2,0.8],[0.22,0.5],[0.25,0.25],[0.45,0.20],[0.60,0.15],[0.9,0.1]]);
    is_non_linear = check_non_linear_trajectory_v1(traj1, thresh = 0.2) 
    print("Traj1 non-linear is", is_non_linear)


    # Assume a non-linear trajectory.
    # But, it is incorrect in this case 
    traj2 = np.array([[0.2,0.0],[0.22,0.22],[0.25,0.25],[0.45,0.25],[0.60,0.20],[0.9,0.0]]);
    is_non_linear = check_non_linear_trajectory_v1(traj2, thresh = 0.2) 
    print("Traj2 non-linear is", is_non_linear)


    # Assume a linear trajectory.
    traj3 = np.array([[0.2,0.0],[0.22,0.0],[0.25,0.0],[0.45,0.0],[0.60,0.0],[0.9,0.0]]);
    is_non_linear = check_non_linear_trajectory_v1(traj3, thresh = 0.2) 
    print("Traj3 non-linear is", is_non_linear)

    # Plot
    plt.figure(1)
    plt.plot(traj1[:,0], traj1[:,1],'ro-')
    plt.axis([-1, 1, -1, 1])
    plt.title("traj1")
    plt.figure(2)
    plt.plot(traj2[:,0], traj2[:,1],'ro-')
    plt.title("traj2")
    plt.axis([-1, 1, -1, 1])
    plt.figure(3)
    plt.plot(traj3[:,0], traj3[:,1],'ro-')
    plt.title("traj2")
    plt.axis([-1, 1, -1, 1])
    #plt.show()


def test_check_non_linear_trajectory_v2():

    # Assume a non-linear trajectory.
    traj1 = np.array([[0.2,0.8],[0.22,0.5],[0.30,0.30],[0.45,0.20],[0.60,0.15],[0.9,0.1]]);
    is_non_linear = check_non_linear_trajectory_v2(traj1, thresh = 0.2) 
    print("Traj1 non-linear is", is_non_linear)

    # Assume a non-linear trajectory.
    # But, it is incorrect in this case 
    traj2 = np.array([[0.2,0.0],[0.22,0.22],[0.25,0.25],[0.45,0.25],[0.60,0.20],[0.9,0.0]]);
    is_non_linear = check_non_linear_trajectory_v2(traj2, thresh = 0.2) 
    print("Traj2 non-linear is", is_non_linear)


    # Assume a linear trajectory.
    traj3 = np.array([[0.2,0.0],[0.22,0.0],[0.25,0.0],[0.45,0.0],[0.60,0.0],[0.9,0.0]]);
    is_non_linear = check_non_linear_trajectory_v2(traj3, thresh = 0.2) 
    print("Traj3 non-linear is", is_non_linear)

    # Plot
    plt.figure(1)
    plt.plot(traj1[:,0], traj1[:,1],'ro-')
    plt.axis([-1, 1, -1, 1])
    plt.title("traj1")
    plt.figure(2)
    plt.plot(traj2[:,0], traj2[:,1],'ro-')
    plt.title("traj2")
    plt.axis([-1, 1, -1, 1])
    plt.figure(3)
    plt.plot(traj3[:,0], traj3[:,1],'ro-')
    plt.title("traj2")
    plt.axis([-1, 1, -1, 1])
    plt.show()


def test_distance_point_to_line():

    # test 1: dist = 2;
    startPt = np.array([0,0]) ; endPt   = np.array([3,0]) ; aPt = np.array([2,2])
    dist = distance_point_to_line(startPt,endPt,aPt)
    print("test 1 dist = " , dist)

    # test 2: dist = 1
    startPt = np.array([-1,1]) ; endPt  = np.array([1,1]) ; aPt = np.array([0,5])
    dist = distance_point_to_line(startPt,endPt,aPt)
    print("test 2 dist = " , dist)

    # test 2: dist = 3
    startPt = np.array([-2,-2]) ; endPt  = np.array([2,-2]) ; aPt = np.array([1,1])
    dist = distance_point_to_line(startPt,endPt,aPt)
    print("test 3 dist = " , dist)


def test_find_grid_cells(): 

    trajectory_list = [[[0.2,0.0],[0.22,0.0],[0.25,0.0],[0.45,0.0],[0.60,0.0],[0.9,0.0]]]
    list_of_grid_id = find_grid_cells(trajectory_list) ; 
    print("list_of_grid_id [36,37,38,39] = ", list_of_grid_id)

    trajectory_list = [[[-0.51,-0.80],[-0.52,-0.60],[-0.60,-0.40],[0.62,0.01],[-0.65,0.26],[-0.70,0.70]]]
    list_of_grid_id = find_grid_cells(trajectory_list) ; 
    print("list_of_grid_id [1,9,17,33,41,49] =", list_of_grid_id)
                      
if __name__ == '__main__':
    #test_distance_point_to_line()
    #test_check_non_linear_trajectory_v1() 
    #test_check_non_linear_trajectory_v2() 
    test_find_grid_cells()
