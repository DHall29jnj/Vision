import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
from config.config import Config


def main():
    # Get the single instance of the Config class
    configs = Config.get_instance()

    # Now you can access all the configuration variables
    print(f"Marker Size: {configs.marker_size}")
    print(f"Camera ID: {configs.camera_id}")
    print(f"STL Path: {configs.stl_path}")
    print(f"ArUco Dictionary: {configs.aruco_dict}")
    print(f"Camera Matrix:\n {configs.camera_matrix}")
    print(f"Marker Size:\n {configs.marker_size}")


if __name__ == "__main__":
    
    main()
    
    # Parameters
    dictionary_id = aruco.DICT_4X4_50
    estimate_pose = True
    show_rejected = False
    camera_id = 0
    video_file = ""  # Leave blank to use webcam

    # Setup dictionary and detector
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)
    
    pntr_id = 1
    ref_id = 2
    marker_size = 0.05

    # Setup video input
    cap = cv2.VideoCapture(video_file if video_file else camera_id)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    wait_time = 1 # milliseconds; 1 allows real-time behavior

    # Define object points for a square planar ArUco marker (z=0)
    obj_points = np.array([
        [-main.configs.marker_size/2,  configs.marker_size/2, 0],
        [ configs.marker_size/2,  configs.marker_size/2, 0],
        [ configs.marker_size/2, -configs.marker_size/2, 0],
        [-configs.marker_size/2, -configs.marker_size/2, 0]
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
            rvecs = []  # Initialize empty lists outside the loop
            tvecs = []  # Initialize empty lists outside the loop
            for i in range(len(ids)):
                marker_id = ids[i] # Access the marker ID
                # Estimate pose for each marker
                ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], configs.cam_matrix, configs.dist_coeffs)
                if ret:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    if marker_id == pntr_id:
                        # Process pointer's pose
                        #print(f"Pointer({pntr_id})pose:")
                        #print("  Rotation:", rvec)
                        #print("  Translation:", tvec)
            
                        # Assume rvec, tvec from solvePnP
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                    
                        # Camera pose in pointer's coordinate system
                        R_cam_to_pntr = rotation_matrix.T  # Inverse rotation
                        t_cam_to_pntr = -R_cam_to_pntr @ tvec  # Inverse translation
                        
                        print("Camera position in pointer's frame:", t_cam_to_pntr)
                        print("Camera orientation in pointer's frame:", R_cam_to_pntr)

                    elif marker_id == ref_id:
                        # Process reference's pose
                        #print(f"Reference({ref_id})pose:")
                        #print("  Rotation:", rvec)
                        #print("  Translation:", tvec)
                        
                        # Assume rvec, tvec from solvePnP
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                    
                        # Camera pose in reference's coordinate system
                        R_cam_to_ref = rotation_matrix.T  # Inverse rotation
                        t_cam_to_ref = -R_cam_to_ref @ tvec  # Inverse translation
                        
                        print("Camera position in the reference frame:", t_cam_to_ref)
                        print("Camera orientation in reference frame:", R_cam_to_ref)
                    cv2.drawFrameAxes(image, configs.cam_matrix, configs.dist_coeffs, rvec, tvec, marker_size)
                    
        # Display the image
        cv2.imshow("ArUco Pose", image)

        # Exit on keypress 'q'
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    
    