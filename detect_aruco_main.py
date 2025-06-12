import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import os

# --- Command-line argument parsing ---
parser = argparse.ArgumentParser(description="ArUco Marker Detection")

parser.add_argument("-d", type=int, default=0,
                    help="Dictionary ID (0=DICT_4X4_50, ..., 10=DICT_6X6_250)")
parser.add_argument("-cd", type=str, help="Path to custom dictionary file (not implemented)")
parser.add_argument("-v", type=str, help="Input video/image file path")
parser.add_argument("-ci", type=int, default=0, help="Camera ID if no video is provided")
parser.add_argument("-c", type=str, help="Camera intrinsic parameters file (YAML)")
parser.add_argument("-l", type=float, default=0.1, help="Marker length in meters")
parser.add_argument("-dp", type=str, help="Path to detector parameters file (YAML)")
parser.add_argument("-r", action="store_true", help="Show rejected candidates")
parser.add_argument("-refine", type=int, help="Corner refinement method (not yet implemented)")

args = parser.parse_args()

# --- Load dictionary ---
dictionary = aruco.getPredefinedDictionary(args.d)
detector_params = aruco.DetectorParameters()

# --- Load detector parameters if provided ---
if args.dp and os.path.isfile(args.dp):
    fs = cv2.FileStorage(args.dp, cv2.FILE_STORAGE_READ)
    detector_params = aruco.DetectorParameters()
    detector_params.readDetectorParameters(fs.root())
    fs.release()

# --- Load camera parameters if pose estimation is needed ---
cam_matrix = np.eye(3, dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
estimate_pose = False

if args.c and os.path.isfile(args.c):
    fs = cv2.FileStorage(args.c, cv2.FILE_STORAGE_READ)
    cam_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    estimate_pose = True

# --- Set coordinate system for marker corners (object points) ---
marker_length = args.l
obj_points = np.array([
    [-marker_length / 2, marker_length / 2, 0],
    [ marker_length / 2, marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32).reshape((4, 1, 3))

# --- Load image or video or start camera ---
input_is_video = False
cap = None
if args.v:
    if os.path.isfile(args.v):
        ext = os.path.splitext(args.v)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            image = cv2.imread(args.v)
        else:
            cap = cv2.VideoCapture(args.v)
            input_is_video = True
    else:
        print(f"File not found: {args.v}")
        exit(1)
else:
    cap = cv2.VideoCapture(args.ci)
    input_is_video = True

# --- Detector ---
detector = aruco.ArucoDetector(dictionary, detector_params)

def process_frame(image):
    corners, ids, rejected = detector.detectMarkers(image)
    image_out = image.copy()

    # Draw detected markers
    if ids is not None:
        aruco.drawDetectedMarkers(image_out, corners, ids)

    # Pose estimation
    if estimate_pose and ids is not None:
        rvecs, tvecs = [], []
        for i in range(len(ids)):
            retval, rvec, tvec = cv2.solvePnP(obj_points, corners[i], cam_matrix, dist_coeffs)
            rvecs.append(rvec)
            tvecs.append(tvec)
            cv2.drawFrameAxes(image_out, cam_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

    if args.r and rejected is not None:
        aruco.drawDetectedMarkers(image_out, rejected, borderColor=(100, 0, 255))

    return image_out

# --- Run on single image or loop over video frames ---
if not input_is_video:
    output = process_frame(image)
    cv2.imshow("ArUco Detection", output)
    cv2.waitKey(0)
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output = process_frame(frame)
        cv2.imshow("ArUco Detection", output)
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):
            break
    cap.release()

cv2.destroyAllWindows()