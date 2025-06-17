import os
import cv2
import cv2.aruco as aruco
import numpy as np
import trimesh
import pyrender
from pathlib import Path
# --- Configuration ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()

# Assume standard 3x3 camera matrix and distortion coefficients
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

# Load STL file
stl_path = "C:/Users/DHall29/workspace/app/app/assets/1121.STL"  # Replace with your STL file path
mesh = trimesh.load(stl_path)

# Convert Trimesh to Pyrender mesh
tri_mesh = pyrender.Mesh.from_trimesh(mesh)
scene = pyrender.Scene()

# Create a camera node
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
camera_node = scene.add(camera)

# Add light
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
scene.add(light)

# Renderer
r = pyrender.OffscreenRenderer(640, 480)

# --- Video Capture Loop ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.ArucoDetector(gray, aruco_dict, aruco_params)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for i, marker_id in enumerate(ids):
            # Draw detected marker
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Get rotation and translation vectors
            R, _ = cv2.Rodrigues(rvecs[i])
            t = tvecs[i]

            # Build transformation matrix
            T = np.identity(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

            # Clear previous scene
            for n in list(scene.mesh_nodes):
                scene.remove_node(n)

            # Add mesh to scene with transformation
            scene.add(tri_mesh, pose=T)

            # Render the scene to image
            color, depth = r.render(scene)

            # Overlay rendered 3D object onto original frame
            mask = (depth != 0)
            frame[mask] = color[mask]

    # Show result
    cv2.imshow("AR Display", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()