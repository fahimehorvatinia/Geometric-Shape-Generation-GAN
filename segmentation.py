# segmentation.py

from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import numpy as np
import os

def segment_shape(file_path, output_path, num_clusters=5):
    points = np.load(file_path)
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(points)
    segmented_points = {i: points[labels == i] for i in range(num_clusters)}
    np.save(output_path, {"hull_vertices": hull_vertices, "clusters": segmented_points})

# Directory paths
input_dir = "3D_Shape_Dataset/Preprocessed"
output_dir = "3D_Shape_Dataset/Segmented"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith("_preprocessed.npy"):
        shape_name = filename.replace("_preprocessed.npy", "")
        segment_shape(
            os.path.join(input_dir, filename),
            os.path.join(output_dir, f"{shape_name}_segmented.npy")
        )
