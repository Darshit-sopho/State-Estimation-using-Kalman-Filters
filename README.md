# c2m5_assignment_file

Final capstone project as a part of online Coursera course "State Estimation and Localization of Self Driving Cars" offered by University of Toronto. This course is the second course in the Self Driving Cars Specialization.

Project Description:

In this assignment, you will implement the Error-State Extended Kalman Filter (ES-EKF) to localize a vehicle using data from the CARLA simulator. The project has three parts, which should be completed in sequence:

First, you will fill in the skeleton implementation of the ES-EKF that is provided, by writing code to perform the filter prediction step and the correction step. The filter relies on IMU data to propagate the state forward in time, and GPS and LIDAR position updates to correct the state estimate. For Part 1 of the project, the sensor data have been prepackaged for you - it is possible to visualize the output of the estimator and compare it to the ground truth vehicle position (the ground truth position data are also provided). Your complete filter implementation will be tested by comparing the estimated vehicle position (produced by your code) with the ground truth position, for a 'hold out' portion of the trajectory (i.e., for which ground truth is not provided to you).

In Part 2, you will examine the effects of sensor miscalibration on the vehicle pose estimates. Specifically, you will uncomment a block of code that intentionally alters the transformation between the LIDAR sensor frame and the IMU sensor frame; use of the incorrect transform will result in errors in the vehicle position estimates. After looking at the errors, your task is to determine how to adjust the filter parameters (noise variances) to attempt to compensate for these errors. The filter code itself should remain unchanged. Your updated filter (with the new parameter(s)) will be tested in the same way as in Part 1.

In Part 3, you will explore the effects of sensor dropout, that is, when all external positioning information (from GPS and LIDAR) is lost for a short period of time. For Part 3, you will load a different dataset where a portion of the GPS and LIDAR measurements are missing (see the detailed instructions below). The goal of Part 3 is to illustrate how the loss of external corrections results in drift in the vehicle position estimate, and also to aid in understanding how the uncertainty in the position estimate changes when sensor measurements are unavailable.
