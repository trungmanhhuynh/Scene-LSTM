# File name: estimateHomographyMatrix.py
# Description: Estimate the homography matrix to convert 
# pixels (camera coordinates) to meter (world coordinate)
# For other datasets, the values of each estimated locations 
# must be changed. 
# Author: Huynh Manh
# Date: 12/19/2018

import cv2
import numpy as np 


def calculate_ucy_homography_matrix():

    # find the homography matrix to convert from camera coordinates (pixels)
    # to world coordinates (meters). The world coordinate has origin
    # at the top-left corner of the car. 
    pts_img = np.array([[476, 117], [562, 117], [562, 311],[476, 311]])     # 4  corners of the car in pixels
    pts_wrd = np.array([[0, 0], [1.81, 0], [1.81, 4.63],[0, 4.63]])         # 4 corners of the car in meters 
    h1, status = cv2.findHomography(pts_img, pts_wrd)                       # homography matrix that convert image -> world

    # Move the origin at top-left corner of the car to bottom-left corner of the location
    # corresponding to the top-left corner of the image. 
    bl_pix = np.array([0,576,1])             # pixel location of bottom-left image 
    bl_meter = h1.dot(bl_pix)                # world-coordinate of bottom-left 
    h2 = np.array([[1, 0, abs(bl_meter[0])],[0, -1 , abs(bl_meter[1])],[0, 0, 1]])

    h = h2.dot(h1)              # the final homography matrix
    return h

def calculate_towncentre_homography_matrix():

    # find the homography matrix to convert from camera coordinates (pixels)
    # to world coordinates (meters). The world coordinate has origin
    # at the top-left corner of the patch on street. 
    pts_img = np.array([[597,702 ], [723, 730], [637, 802],[512, 775]])     # 4  corners of the car in pixels
    pts_wrd = np.array([[0, 0], [1, 0], [1, 1],[0, 1]])                  # 4 corners of the car in meters 
    h1, status = cv2.findHomography(pts_img, pts_wrd)                            # homography matrix that convert image -> world

    # Move the origin at top-left corner of the car to bottom-left corner of the location
    # corresponding to the top-left corner of the image. 
    bl_pix = np.array([0,1080,1])             # pixel location of bottom-left image 
    bl_meter = h1.dot(bl_pix)                # world-coordinate of bottom-left 
    h2 = np.array([[1, 0, abs(bl_meter[0])],[0, -1 ,abs(bl_meter[1])],[0, 0, 1]])

    h = h2.dot(h1)              # the final homography matrix
    return h

def calculate_PETS09_homography_matrix():

    # find the homography matrix to convert from camera coordinates (pixels)
    # to world coordinates (meters). The world coordinate has origin
    # at the top-left corner of the patch on street. 
    pts_img = np.array([[597,702 ], [723, 730], [637, 802],[512, 775]])     # 4  corners of the car in pixels
    pts_wrd = np.array([[0, 0], [1, 0], [1, 1],[0, 1]])                  # 4 corners of the car in meters 
    h1, status = cv2.findHomography(pts_img, pts_wrd)                            # homography matrix that convert image -> world

    # Move the origin at top-left corner of the car to bottom-left corner of the location
    # corresponding to the top-left corner of the image. 
    bl_pix = np.array([0,576,1])             # pixel location of bottom-left image 
    bl_meter = h1.dot(bl_pix)                # world-coordinate of bottom-left 
    h2 = np.array([[1, 0, abs(bl_meter[0])],[0, -1 , abs(bl_meter[1])],[0, 0, 1]])

    h = h2.dot(h1)              # the final homography matrix
    return h


def calculate_grandcentral_homography_matrix():

    # find the homography matrix to convert from camera coordinates (pixels)
    # to world coordinates (meters). The world coordinate has origin
    # at the top-left corner of the patch on street. 
    pts_img = np.array([[302, 218], [379, 218], [386, 306],[303, 308]])     # 4  corners of the car in pixels
    pts_wrd = np.array([[0, 0], [5, 0], [5, 5],[0, 5]])                  # 4 corners of the car in meters 
    h1, status = cv2.findHomography(pts_img, pts_wrd)                            # homography matrix that convert image -> world

    # Move the origin at top-left corner of the car to bottom-left corner of the location
    # corresponding to the top-left corner of the image. 
    bl_pix = np.array([0,480,1])             # pixel location of bottom-left image 
    bl_meter = h1.dot(bl_pix)                # world-coordinate of bottom-left 
    h2 = np.array([[1, 0, abs(bl_meter[0])],[0, -1 , abs(bl_meter[1])],[0, 0, 1]])

    h = h2.dot(h1)              # the final homography matrix
    return h

def test_ucy_homography_matrix(h):

    print("the transformation matrix of UCY dataset is:")
    print(h)

    print("testing pixel (0,576,1), it should be [0,0,1] in world-coordinate"); 
    test_pt1_pix = np.array([0,576,1])
    test_pt1_met = h.dot(test_pt1_pix)

    print("test_pt1_pix =", test_pt1_pix)
    print("test_pt1_met =", test_pt1_met)

    print("testing pixel (206,268,1), it should be [4.33558125829, 7.35072183117,1] in world-coordinate \
           same data used in SGAN (data from zara_01, frame 1, human id 8"); 
    test_pt2_pix = np.array([206,268,1])
    test_pt2_met = h.dot(test_pt2_pix)

    print("test_pt2_pix =", test_pt2_pix)
    print("test_pt2_met =", test_pt2_met)


def test_towncentre_homography_matrix(h):

    print("the transformation matrix of town centre dataset is:")
    print(h)

    print("testing pixel (0,1080,1), it should be [0,0,1] in world-coordinate"); 
    test_pt1_pix = np.array([0,1080,1])
    test_pt1_met = h.dot(test_pt1_pix)
    print("test_pt1_pix =", test_pt1_pix)
    print("test_pt1_met =", test_pt1_met)

def test_grandcentral_homography_matrix(h):

    print("the transformation matrix of grandcentral dataset is:")
    print(h)

    print("testing pixel (0,480,1), it should be [0,0,1] in world-coordinate"); 
    test_pt1_pix = np.array([0,480,1])
    test_pt1_met = h.dot(test_pt1_pix)
    print("test_pt1_pix =", test_pt1_pix)
    print("test_pt1_met =", test_pt1_met)

if __name__ == '__main__':
    h = calculate_grandcentral_homography_matrix()
    test_grandcentral_homography_matrix(h)