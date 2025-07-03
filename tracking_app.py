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
marker_size = 0.2  # In meters

# Pointer offset (3D vector from marker center to pointer tip)
# You need to determine this through calibration (e.g., by touching known points)
pointer_offset = np.array([0, 0, 0.1], dtype=np.float32) # Example: 10cm along the marker's Z-axis

# Parameters
dictionary_id = aruco.DICT_6X6_250
estimated_pose = True
show_rejected = False
camera_id = 0
video_file = "" #Leave blank to use the webcam

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

# File to save pointer coordinates
output_file = "pointer_coordinates.txt"

#List all tracked pointer positions 2D
tracked_points = []
_, frame = cap.read() #read one frame to get image dimensions
trajectory_canvas = np.zeros_like(frame)

previous_pointer_pos_img = None

previous_filtered_pos = None

# Store pose data
pointer_poses = []

while True:
    ret, image = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    current_pointer_pos_img = None
    
    if not ids:
        #print("References on bone and/or pointer were not detected.")
        continue
    
    # if not 63 in ids or not 23 in ids:
    #     print("")
    #     continue
    for id in ids:
        print(f"id {id}")
    # print("We have the markers!")
    
    
    
    
      
#         """ pointer_rvec = None
#         pointer_tvec = None
#         reference_rvec = None
#         reference_tvec = None
#         for i in range(len(ids)):
#             # Assuming marker ID 23 is the pointer marker and ID 62 is the reference marker
#             if ids[i] == 23:
#                 ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], cam_matrix, dist_coeffs)
#                 if ret:
#                     pointer_rvec = rvec
#                     pointer_tvec = tvec
#                     # Draw axes for pointer marker
#                     cv2.drawFrameAxes(image, cam_matrix, dist_coeffs, rvec, tvec, marker_size)
#             elif ids[i] == 62:
#                 ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], cam_matrix, dist_coeffs)
#                 if ret:
#                     reference_rvec = rvec
#                     reference_tvec = tvec
#                     # Draw axes for reference marker
#                     cv2.drawFrameAxes(image, cam_matrix, dist_coeffs, rvec, tvec, marker_size * 2) # Larger axes for ref marker

#         if pointer_rvec is not None and reference_rvec is not None:
#             # Calculate pointer tip position in camera frame
#             pointer_tip_camera_frame = cv2.Rodrigues(pointer_rvec) [0] @ pointer_offset + pointer_tvec

#             # Calculate transformation from camera frame to reference frame
#             # This is the inverse of the transformation from reference frame to camera frame
#             R_ref_to_cam, _ = cv2.Rodrigues(reference_rvec)
#             t_ref_to_cam = reference_tvec

#             R_cam_to_ref = R_ref_to_cam.T # Transpose of rotation matrix is its inverse
#             t_cam_to_ref = -R_cam_to_ref @ t_ref_to_cam

#             # Transform pointer tip position from camera frame to reference frame
#             pointer_tip_reference_frame = R_cam_to_ref @ pointer_tip_camera_frame + t_cam_to_ref

#             # Save the coordinates
#             with open(output_file, "a") as f:
#                 f.write(f"{pointer_tip_reference_frame[0,0]} {pointer_tip_reference_frame[1,0]} {pointer_tip_reference_frame[2,0]}\n")
            
#             # Print the coordinates (optional)
#             # print(f"Pointer tip in reference frame: {pointer_tip_reference_frame}")

#             # Project pointer tip position in camera frame to 2D image plane
#             img_points, _ = cv2.projectPoints(pointer_tip_camera_frame, np.zeros((3,1)), np.zeros((3,1)), cam_matrix, dist_coeffs)
#             if img_points is not None:
#                 current_pointer_pos_img = (int(img_points[0][0][0]), int(img_points[0][0][1]))

#                 #Append the current position to the list of points
#                 tracked_points.append(current_pointer_pos_img)
#                 if len(tracked_points) > 1:
#                     pts = np.array(tracked_points, np.int32).reshape((-1, 1, 2))
#                     cv2.polylines(trajectory_canvas, [pts], False, (0, 0, 255), 2)

#                 # Draw a circle at the current pointer tip position
#                 cv2.circle(trajectory_canvas, current_pointer_pos_img, 5, (0, 255, 0), -1) # Green circle

#         aruco.drawDetectedMarkers(image, corners, ids) # Draw detected markers and their IDs

#     #Overlay the trajectory canvas onto the image
#     combined_image = cv2.addWeighted(image, 1, trajectory_canvas, 1, 0)

#     # Display the image
#     cv2.imshow("ArUco Pointer Tracking", combined_image)

#     # Exit on keypress 'q'
#     if cv2.waitKey(wait_time) & 0xFF == ord('q'):
#         break

# # Release the video capture object and destroy all windows
# cap.release()
# cv2.destroyAllWindows() """