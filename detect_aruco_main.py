import cv2
import cv2.aruco as aruco
import numpy as np
import time

# Parameters (adjust these as needed or load from command line)
dictionary_id = aruco.DICT_6X6_250
marker_length = 0.05  # In meters (used for pose estimation)
estimate_pose = True
show_rejected = True
camera_id = 0
video_file = ""  # Leave blank to use webcam



# Setup dictionary and detector
dictionary = aruco.getPredefinedDictionary(dictionary_id)
detector_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detector_params)

# Setup video input
cap = cv2.VideoCapture(video_file if video_file else camera_id)
wait_time = 0 if video_file else 10

# Set coordinate system
obj_points = np.array([
    [-marker_length/2,  marker_length/2, 0],
    [ marker_length/2,  marker_length/2, 0],
    [ marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32).reshape((4, 1, 3))

total_time = 0
total_iterations = 0

while cap.isOpened():
    ret = cap.grab()
    if not ret:
        break

    ret, image = cap.retrieve()
    if not ret:
        continue

    start_tick = cv2.getTickCount()

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(image)

    # Estimate pose
    rvecs, tvecs = [], []
    if estimate_pose and ids is not None:
        for c in corners:
            retval, rvec, tvec = cv2.solvePnP(obj_points, c, cam_matrix, dist_coeffs)
            rvecs.append(rvec)
            tvecs.append(tvec)

    # Time stats
    elapsed = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
    total_time += elapsed
    total_iterations += 1

    if total_iterations % 30 == 0:
        print(f"Detection Time = {elapsed * 1000:.2f} ms "
              f"(Mean = {1000 * total_time / total_iterations:.2f} ms)")

    # Draw output
    image_copy = image.copy()
    if ids is not None:
        aruco.drawDetectedMarkers(image_copy, corners, ids)
        if estimate_pose:
            for i in range(len(ids)):
                cv2.drawFrameAxes(image_copy, cam_matrix, dist_coeffs,
                                  rvecs[i], tvecs[i], marker_length * 1.5, 2)

    if show_rejected and rejected is not None:
        aruco.drawDetectedMarkers(image_copy, rejected, borderColor=(100, 0, 255))

    cv2.imshow("out", image_copy)
    if cv2.waitKey(wait_time) == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()