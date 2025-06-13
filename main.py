import cv2
import cv2.aruco as aruco
import numpy as np

if __name__ == "__main__":
    # TODO: It should be derived from the calibration procedure and parsed by using yml file    
    # Camera matrix and distortion coefficients (example values)    
    cam_matrix = np.array([[800, 0, 320],
                           [0, 800, 240],
                           [0, 0, 1]], dtype=np.float32)

    dist_coeffs = np.array([-0.3, 0.12, 0.001, 0.002, 0.0], dtype=np.float32)

    # Printed ARuco dimensions
    marker_size = 0.05  # In meters (used for pose estimation)

    # Parameters
    dictionary_id = aruco.DICT_6X6_250
    estimate_pose = True
    show_rejected = False
    camera_id = 0
    video_file = ""  # Leave blank to use webcam

    # Setup dictionary and detector
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)

    # Setup video input
    cap = cv2.VideoCapture(video_file if video_file else camera_id)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    wait_time = 1 # milliseconds; 1 allows real-time behavior

    # Define object points for a square planar ArUco marker (z=0)
    obj_points = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)

    while True:
        ret, image = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # Question: Are you planning to use the estimated pose for any further processing?
            # If yes, not that the rvecs and tvecs are stored in lists, but they are not avaliable outside the while loop
            # How many frames you want to store inside the lists?
            rvecs = []
            tvecs = []
            for i in range(len(ids)):
                # Estimate pose using solvePnP
                ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], cam_matrix, dist_coeffs)

                if ret: # Check if solvePnP was successful
                    rvecs.append(rvec)
                    tvecs.append(tvec)

                    # Draw detected marker and axes
                    aruco.drawDetectedMarkers(image, corners) # Draw the detected markers
                    cv2.drawFrameAxes(image, cam_matrix, dist_coeffs, rvec, tvec, marker_size) # Draw the axes

        # Display the image
        cv2.imshow("ArUco Pose", image)

        # Exit on keypress 'q'
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    
    # Question: implement a method to store the rvecs and tvecs in a file

        

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()