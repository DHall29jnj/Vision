import cv2
import cv2.aruco as aruco
import numpy as np
import dotenv
import os

dotenv.load_dotenv()

# TODO: Improve it

class Config:
    def __init__(self):

        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)        
        self.aruco_params = aruco.DetectorParameters()

        self.fx = 800
        self.fy = 800
        self.cx = 320
        self.cy = 240

        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                       [0, self.fy, self.cy],
                                       [0, 0, 1]], dtype=np.float32)

        self.dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

        self.marker_size = 0.05  # In meters (used for pose estimation)

        self.marker_id = 23


        self.camera_id = int(os.getenv("CAMERA_ID", 0))
        self.assets_folder = os.getenv("ASSETS_FOLDER", "assets")
        self.model = os.getenv("MODEL", "1130.STL")
        self.stl_path = os.path.join(self.assets_folder, self.model)

        if not os.path.exists(self.stl_path):
            raise ValueError(f"STL file not found: {self.stl_path}")
        

    @staticmethod
    def get_instance():
        if not hasattr(Config, "instance"):
            Config.instance = Config()
        return Config.instance
        