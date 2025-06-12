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

    wait_time = 1  # milliseconds; 1 allows real-time behavior

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
            # Estimate pose
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, cam_matrix, dist_coeffs)

            for i in range(len(ids)):
                # Draw detected marker and axes
                aruco.drawDetectedMarkers(image, corners, ids)
                aruco.drawAxis(image, cam_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)

                # Optional: Put text near each marker
                c = corners[i][0]
                x = int(c[:, 0].mean())
                y = int(c[:, 1].mean())
                cv2.putText(image, f"ID: {ids[i][0]}", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show output
        cv2.imshow("ArUco Pose", image)

        # Exit on ESC or 'q'
        key = cv2.waitKey(wait_time)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    #     # Estimate pose
    #     rvecs, tvecs = [], []
    #     if estimate_pose and ids is not None:
    #         for c in corners:
    #             retval, rvec, tvec = cv2.solvePnP(obj_points, c, cam_matrix, dist_coeffs)
    #             rvecs.append(rvec)
    #             tvecs.append(tvec)

    #     # Time stats
    #     elapsed = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
    #     total_time += elapsed
    #     total_iterations += 1

    #     if total_iterations % 30 == 0:
    #         print(f"Detection Time = {elapsed * 1000:.2f} ms "
    #             f"(Mean = {1000 * total_time / total_iterations:.2f} ms)")

    #     # Draw output
    #     image_copy = image.copy()
    #     if ids is not None:
    #         aruco.drawDetectedMarkers(image_copy, corners, ids)
    #         if estimate_pose:
    #             for i in range(len(ids)):
    #                 cv2.drawFrameAxes(image_copy, cam_matrix, dist_coeffs,
    #                                 rvecs[i], tvecs[i], marker_size * 1.5, 2)

    #     if show_rejected and rejected is not None:
    #         aruco.drawDetectedMarkers(image_copy, rejected, borderColor=(100, 0, 255))

    #     cv2.imshow("out", image_copy)
    #     if cv2.waitKey(wait_time) == 27:  # Esc key to exit
    #         break

    # cap.release()
    # cv2.destroyAllWindows()