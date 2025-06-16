import cv2
import numpy as np
import trimesh
import pyrender

from app.core.config import Config

if __name__ == "__main__":
    config = Config.get_instance()
    
    # --- Configuration ---
    aruco_dict = config.aruco_dict
    aruco_params = config.aruco_params

    # Assume standard 3x3 camera matrix and distortion coefficients
    camera_matrix = config.camera_matrix
    dist_coeffs = config.dist_coeffs

    # Load STL file
    stl_path = config.stl_path
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

        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # For each detected marker
            for i, marker_id in enumerate(ids):
                # Draw detected marker
                cv2.aruco.drawDetectedMarkers(frame, corners)

                # Get marker corners (4 corners per marker)
                marker_corners = corners[i][0]
                
                # Create object points (3D coordinates of marker corners)
                obj_points = np.array([
                    [-config.marker_size/2, -config.marker_size/2, 0],
                    [ config.marker_size/2, -config.marker_size/2, 0],
                    [ config.marker_size/2,  config.marker_size/2, 0],
                    [-config.marker_size/2,  config.marker_size/2, 0]
                ], dtype=np.float32)

                # Solve for pose
                success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners, camera_matrix, dist_coeffs)
                
                if success:
                    # Get rotation and translation vectors
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec

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
