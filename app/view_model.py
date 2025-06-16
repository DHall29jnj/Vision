import trimesh
import pyrender
import numpy as np
from app.core.config import Config

if __name__ == "__main__":
    config = Config.get_instance()

    # Load mesh
    mesh = trimesh.load(config.stl_path)
    mesh.apply_scale(0.1)  # Adjust scale
    mesh.fix_normals()

    # Create scene
    scene = pyrender.Scene()
    tri_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(tri_mesh)

    # Add light and camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=np.eye(4))

    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light)

    # Show viewer
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)