The cv2.solvePnP() function estimates the 3D pose of the object (marker) relative to the camera,
 
 
import cv2
import numpy as np
 
# Assume rvec, tvec from solvePnP
rotation_matrix, _ = cv2.Rodrigues(rvec)
 
# Camera pose in marker's coordinate system
R_cam_to_marker = rotation_matrix.T  # Inverse rotation
t_cam_to_marker = -R_cam_to_marker @ tvec  # Inverse translation
 
print("Camera position in marker's frame:", t_cam_to_marker)
print("Camera orientation in marker's frame:", R_cam_to_marker)