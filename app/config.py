import cv2
import cv2.aruco as aruco
import numpy as np
import os
import yaml


try:
        with open('calibration_params.yml', 'r') as f:
            calibration_data = yaml.safe_load(f)

        cam_matrix = np.array(calibration_data['camera_matrix'], dtype=np.float32)
        disto_coeffs = np.array(calibration_data['dist_coeff'], dtype=np.float32)

        print("Camera calibration parameters loaded successfully.")
        print("Camera matrix:\n", cam_matrix)
        print("Distortion coefficients:\n", disto_coeffs)

except FileNotFoundError:
        print("Error: calibration_params.yml not found. Please run the calibration script first.")
        exit()
except yaml.YAMLError as e:
        print("Error loading YAML file:", e)
        exit()

class Config:
    def __init__(self):

        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)        
        self.aruco_params = aruco.DetectorParameters()

        self.camera_matrix = cam_matrix

        self.dist_coeffs = disto_coeffs

        self.marker_size = 0.025  # In meters (used for pose estimation)

        self.pointer_id = 62
        self.stat_ref_id = 23


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
        