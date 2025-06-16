import cv2
import numpy as np
import trimesh
import pyrender
from app.core.config import Config

if __name__ == "__main__":
    config = Config.get_instance()

    # --- ArUco Setup ---
    aruco_dict = config.aruco_dict
    aruco_params = config.aruco_params

    camera_matrix = config.camera_matrix
    dist_coeffs = config.dist_coeffs

    # Load STL file
    stl_path = config.stl_path
    mesh = trimesh.load(stl_path)

    # Center and normalize mesh
    mesh.apply_translation(-mesh.centroid)  # center it
    scale_factor = 1.0 / mesh.scale  # Normalize size
    mesh.apply_scale(scale_factor * 0.7)  # Make it AR-friendly

    # Use a PBR material with color for visibility
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        roughnessFactor=0.8,
        baseColorFactor=[0.3, 0.6, 0.9, 1.0]  # Bright blueish color
    )
    tri_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    # Scene setup
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])  # Transparent background

    # Add light
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light)

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=640/480)
    camera_pose = np.eye(4)
    scene.add(camera, pose=camera_pose)

    # Renderer
    r = pyrender.OffscreenRenderer(640, 480)

    # Video Capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            for i, marker_id in enumerate(ids):
                cv2.aruco.drawDetectedMarkers(frame, corners)

                marker_corners = corners[i][0]

                obj_points = np.array([
                    [-config.marker_size/2, -config.marker_size/2, 0],
                    [ config.marker_size/2, -config.marker_size/2, 0],
                    [ config.marker_size/2,  config.marker_size/2, 0],
                    [-config.marker_size/2,  config.marker_size/2, 0]
                ], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners, camera_matrix, dist_coeffs)

                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec

                    T = np.identity(4)
                    T[:3, :3] = R
                    T[:3, 3] = t.flatten()

                    # Remove old mesh
                    for n in list(scene.mesh_nodes):
                        scene.remove_node(n)

                    scene.add(tri_mesh, pose=T)

                    # Render
                    color, depth = r.render(scene)

                    # Blend render over original frame
                    mask = depth != 0
                    frame[mask] = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)[mask]

        cv2.imshow("AR Display", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()