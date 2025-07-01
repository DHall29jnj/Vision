import cv2
import cv2.aruco as aruco
import numpy as np
from app.core.config import Config

if __name__ == "__main__":    
    config = Config.get_instance()

    # Initialize the point position relative to the marker
    point_pos = np.array([0.0, 0.0, 0.0])  # x, y, z in marker coordinates
    step_size = 0.05  # How much to move with each key press

    # ArUco marker setup
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # Camera calibration parameters - you need to replace these with your actual calibration data
    camera_matrix = config.camera_matrix
    dist_coeffs = config.dist_coeffs

    # Camera setup
    cap = cv2.VideoCapture(0)  # Change to video file path if needed

    # Define marker corner points in object space (assuming marker size is 0.1m)
    marker_size = config.marker_size
    obj_points = np.array([[-marker_size/2, marker_size/2, 0],
                          [marker_size/2, marker_size/2, 0],
                          [marker_size/2, -marker_size/2, 0],
                          [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)
        
        if ids is not None:
            # Draw all detected markers
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Find the first marker with ID 0
            marker_id = config.marker_id  # The ID of the marker we're using as reference
            if marker_id in ids:
                idx = np.where(ids == marker_id)[0][0]
                marker_corners = corners[idx][0]
                
                # Estimate pose using solvePnP
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    marker_corners, 
                    camera_matrix, 
                    dist_coeffs
                )
                
                if success:
                    # Draw coordinate axes for the marker
                    frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size/2)
                    
                    # Convert the point from marker coordinates to camera coordinates
                    rotation_mat, _ = cv2.Rodrigues(rvec)
                    point_cam = rotation_mat @ point_pos + tvec.reshape(3)
                    
                    # Project the 3D point to 2D image coordinates
                    point_2d, _ = cv2.projectPoints(
                        point_pos.reshape(1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs
                    )
                    point_2d = tuple(map(int, point_2d.reshape(-1, 2)[0]))
                    
                    # Draw the point as a red circle
                    cv2.circle(frame, point_2d, 5, (0, 0, 255), -1)
                    
                    # Display the point's coordinates
                    cv2.putText(frame, f"Point: {point_pos[0]:.2f}, {point_pos[1]:.2f}, {point_pos[2]:.2f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Arrow keys: Move XY | WS: Move Z | ESC: Quit", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('ArUco Marker Tracking', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == 82:  # Up arrow
            point_pos[1] += step_size  # Increase Y
        elif key == 84:  # Down arrow
            point_pos[1] -= step_size  # Decrease Y
        elif key == 81:  # Left arrow
            point_pos[0] -= step_size  # Decrease X
        elif key == 83:  # Right arrow
            point_pos[0] += step_size  # Increase X
        elif key == ord('w'):  # 'w' key for Z+
            point_pos[2] += step_size
        elif key == ord('s'):  # 's' key for Z-
            point_pos[2] -= step_size

    cap.release()
    cv2.destroyAllWindows()