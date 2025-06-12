import cv2
import numpy as np

def find_chessboard_corners(images, pattern_size):
    obj_points = []
    img_points = []

    # Prepare the 3D points of the chessboard corners (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    return obj_points, img_points

def calibrate_camera(obj_points, img_points, img_size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    return ret, mtx, dist, rvecs, tvecs

def undistort_image(img, mtx, dist):
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted_img


import glob

# Load the images
image_files = glob.glob('C:/Users/DHall29/workspace/Vision/images/*.jpg')
images = [cv2.imread(file) for file in image_files]

# Find the chessboard corners
pattern_size = (9, 6)
obj_points, img_points = find_chessboard_corners(images, pattern_size)

# Calibrate the camera
img_size = (images[0].shape[1], images[0].shape[0])
ret, mtx, dist, rvecs, tvecs = calibrate_camera(obj_points, img_points, img_size)

# Undistort an example image
undistorted_example = undistort_image(images[0], mtx, dist)

# Show the original and undistorted images
cv2.imshow('Original Image', images[0])
cv2.imshow('Undistorted Image', undistorted_example)

cv2.waitKey(0)
cv2.destroyAllWindows()