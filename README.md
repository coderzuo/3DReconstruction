# 3DReconstruction
Stereo reconstruction using OpenCV 3.4.4 and python 3.4

This Repository contains 2 scripts in two folders. 

The first folder called "Calibration" contains the script called calibrate.py

Usage for this script is:

$python calibrate.py

This script iteratively reads the pictures contained in the folder calibration_images and calculates the intrinsic parameters of the camera used to take those pictures. 

These parameters are saved as a numpy array in the folder camera_params. 

The second folder called "Reconstruction" contains the script called disparity.py

This script opens a pair of images from the folder "reconstruct_this" and generates a point cloud (.ply file). 

Usage for this script is:

$python disparity.py

An in depth explanation of the code and its intended use can be found at: 
https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-i-c013907d1ab5

