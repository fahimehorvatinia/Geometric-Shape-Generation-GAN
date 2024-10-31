# preprocessing.py

import numpy as np
from scipy.ndimage import gaussian_filter
import os

def preprocess_shape(file_path, output_path):
    points = np.load(file_path)
    max_val = np.max(np.abs(points))
    normalized_points = points / max_val
    smoothed_points = gaussian_filter(normalized_points, sigma=0.1)
    np.save(output_path, smoothed_points)

# Directory for the shapes
input_dir = "3D_Shape_Dataset"
output_dir = "3D_Shape_Dataset/Preprocessed"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each shape file
for filename in os.listdir(input_dir):
    if filename.endswith("_points.npy"):
        shape_name = filename.replace("_points.npy", "")
        preprocess_shape(
            os.path.join(input_dir, filename),
            os.path.join(output_dir, f"{shape_name}_preprocessed.npy")
        )
