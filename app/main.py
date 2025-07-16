import cv2
import cv2.aruco as aruco
import numpy as np
import trimesh
from core.config import Config

if __name__ == "__main__":
    config = Config.get_instance()
    
    # --- ArUco Setup ---
    aruco_dict = config.aruco_dict
    aruco_params = config.aruco_params

    cam_matrix = config.cam_matrix
    dist_coeffs = config.dist_coeffs
    pntr_id = config.pntr_id
    ref_id = config.ref_id
    marker_size = config.marker_size
    
    # Parameters
    estimate_pose = True
    show_rejected = False
    camera_id = config.camera_id
    video_file = ""  # Leave blank to use webcam

    # Setup dictionary and detector
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    
    
    # Setup video input
    cap = cv2.VideoCapture(video_file if video_file else camera_id)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    wait_time = 1 # milliseconds; 1 allows real-time behavior

    # Define object points for a square planar ArUco marker (z=0)
    obj_points = np.array([
        [-marker_size/2, marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)

    # Variables to store the pose of the reference and pointer markers
    rvec_ref, tvec_ref = None, None
    rvec_pntr, tvec_pntr = None, None
    
    
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
                # Reset pose information at the start of each frame
            rvec_ref, tvec_ref = None, None
            rvec_pntr, tvec_pntr = None, None
            for i in range(len(ids)):
                marker_id = ids[i] # Access the marker ID
                # Calculate the pose for each marker
                ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], cam_matrix, dist_coeffs)
                if ret:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    
                    if marker_id == ref_id:
                        rvec_ref = rvec
                        tvec_ref = tvec
                    elif marker_id == pntr_id:
                        rvec_pntr = rvec
                        tvec_pntr = tvec
                    
                # Display the marker ID
                    center_x = int(np.mean(corners[i][0][:, 0]))
                    center_y = int(np.mean(corners[i][0][:, 1])) - 10

                    # Define font, scale, color, and thickness
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.75
                    font_color = (0, 255, 255) # Yellow
                    font_thickness = 2

                    # Convert the marker_id to a string
                    text = f"   ID: {marker_id}"

                    # Put the text on the image
                    cv2.putText(image, text, (center_x, center_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                    #End of marker ID display
                    
                    if marker_id == pntr_id:
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        # Camera pose in pointer's coordinate system
                        R_cam_to_pntr = rotation_matrix.T  # Inverse rotation
                        t_cam_to_pntr = -R_cam_to_pntr @ tvec  # Inverse translation
                        # print("Camera position in pointer's frame:", t_cam_to_pntr)
                        # print("Camera orientation in pointer's frame:", R_cam_to_pntr)

                    elif marker_id == ref_id:
                        # Process reference's pose
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        # Camera pose in reference's coordinate system
                        R_cam_to_ref = rotation_matrix.T  # Inverse rotation
                        t_cam_to_ref = -R_cam_to_ref @ tvec  # Inverse translation
                        # print("Camera position in the reference frame:", t_cam_to_ref)
                        # print("Camera orientation in reference frame:", R_cam_to_ref)
                    cv2.drawFrameAxes(image, cam_matrix, dist_coeffs, rvec, tvec, marker_size)
                    
            if rvec_ref is not None and tvec_ref is not None and rvec_pntr is not None and tvec_pntr is not None:
                # 1. Convert rvec and tvec to homogeneous transformation matrices
                R_ref, _ = cv2.Rodrigues(rvec_ref)
                T_ref = np.eye(4)
                T_ref[:3, :3] = R_ref
                T_ref[:3, 3] = tvec_ref.flatten()

                R_pntr, _ = cv2.Rodrigues(rvec_pntr)
                T_pntr = np.eye(4)
                T_pntr[:3, :3] = R_pntr
                T_pntr[:3, 3] = tvec_pntr.flatten()

                # 2. Invert the reference transformation matrix
                T_ref_inverse = np.linalg.inv(T_ref)

                # 3. Concatenate the transformations to get pointer's pose relative to reference
                T_ref_to_pntr = T_ref_inverse @ T_pntr

                # You can extract R_ref_to_pntr and t_ref_to_pntr if needed
                R_ref_to_pntr = T_ref_to_pntr[:3, :3]
                t_ref_to_pntr = T_ref_to_pntr[:3, 3]

                # You can print or visualize this information Print pointer's position (x, y, z) relative to the reference
                print(f"Pointer relative to Reference ({ref_id} -> {pntr_id}):")
                print("  Translation:", t_ref_to_pntr)

                # Example: Display the relative translation on the image
                rel_pose_text = f"Rel T: X={t_ref_to_pntr[0]:.2f} Y={t_ref_to_pntr[1]:.2f} Z={t_ref_to_pntr[2]:.2f}"
                cv2.putText(image, rel_pose_text, (50, 50), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw the detected markers and their axes for visualization on the text_overlay
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            for i in range(len(ids)):
                # For simplicity, using the current loop's rvec, tvec here
                if ret: # Only draw axes if solvePnP was successful for this marker
                    cv2.drawFrameAxes(image, cam_matrix, dist_coeffs, rvec, tvec, marker_size * 0.5)
            
                    
        # Display the image
        cv2.imshow("ArUco Pose", image)

        # Exit on keypress 'q'
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    
    