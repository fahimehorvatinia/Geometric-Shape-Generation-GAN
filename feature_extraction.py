# feature_extraction.py

import numpy as np
import os

def compute_features(file_path, output_path):
    data = np.load(file_path, allow_pickle=True).item()
    hull_vertices = data["hull_vertices"]
    clusters = data["clusters"]
    normals = np.random.randn(hull_vertices.shape[0], 3)  # Replace with actual normal calculation
    np.save(output_path, {"hull_vertices": hull_vertices, "normals": normals, "clusters": clusters})

# Directory paths
input_dir = "3D_Shape_Dataset/Segmented"
output_dir = "3D_Shape_Dataset/Features"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith("_segmented.npy"):
        shape_name = filename.replace("_segmented.npy", "")
        compute_features(
            os.path.join(input_dir, filename),
            os.path.join(output_dir, f"{shape_name}_features.npy")
        )
