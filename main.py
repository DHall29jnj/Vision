import cv2
import cv2.aruco as aruco
import numpy as np
import yaml

if __name__ == "__main__":
    # Load camera matrix and distortion coefficients from the YAML file
    try:
        with open('calibration_params.yml', 'r') as f:
            calibration_data = yaml.safe_load(f)

        cam_matrix = np.array(calibration_data['camera_matrix'], dtype=np.float32)
        dist_coeffs = np.array(calibration_data['dist_coeff'], dtype=np.float32)

        print("Camera calibration parameters loaded successfully.")
        print("Camera matrix:\n", cam_matrix)
        print("Distortion coefficients:\n", dist_coeffs)

    except FileNotFoundError:
        print("Error: calibration_params.yml not found. Please run the calibration script first.")
        exit()
    except yaml.YAMLError as e:
        print("Error loading YAML file:", e)
        exit()

    # Printed ARuco dimensions
    marker_size = 0.02  # In meters

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
            
            # How many frames you want to store inside the lists?
            max_frames = 50
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
                    
                    RotAndTrans_data = {
                        'rvecs': rvec.tolist(),
                        'tvecs': tvec.tolist()
                    }

                    with open('RotAndTrans_data.yml', 'w') as f:
                        yaml.dump(RotAndTrans_data, f)
                        
                    

                    print("Rotation and Translation vectors  saved to RotAndTrans_data.yml")
                    print("Rotation Vector:\n", rvec)
                    print("Translation Vector:\n", tvec)

        # Display the image
        cv2.imshow("ArUco Pose", image)

        # Exit on keypress 'q'
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()