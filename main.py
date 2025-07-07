import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
from app.config import Config

if __name__ == "__main__":

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
        [-Config.marker_size/2,  Config.marker_size/2, 0],
        [ Config.marker_size/2,  Config.marker_size/2, 0],
        [ Config.marker_size/2, -Config.marker_size/2, 0],
        [-Config.marker_size/2, -Config.marker_size/2, 0]
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
                ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], Config.camera_matrix, Config.dist_coeffs)

                if ret: # Check if solvePnP was successful
                    rvecs.append(rvec)
                    tvecs.append(tvec)

                    # Draw detected marker and axes
                    aruco.drawDetectedMarkers(image, corners) # Draw the detected markers
                    cv2.drawFrameAxes(image, Config.camera_matrix, Config.dist_coeffs, rvec, tvec, Config.marker_size) # Draw the axes
                    
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