import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_point_cloud(points, title, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    plt.savefig(output_path)
    plt.close()

# Load your point cloud data (example for 'Cone')
original_points = np.load("3D_Shape_Dataset/Cone_points.npy")
preprocessed_points = np.load("3D_Shape_Dataset/Preprocessed/Cone_preprocessed.npy")
segmented_points = np.load("3D_Shape_Dataset/Segmented/Cone_segmented.npy", allow_pickle=True).item()
feature_points = np.load("3D_Shape_Dataset/Features/Cone_features.npy", allow_pickle=True).item()

# Generate illustrations
plot_point_cloud(original_points, "Original Cone Shape", "illustrations/original_cone.png")
plot_point_cloud(preprocessed_points, "Preprocessed Cone Shape", "illustrations/preprocessed_cone.png")
plot_point_cloud(segmented_points["clusters"][0], "Segmented Cone Shape - Cluster 1", "illustrations/segmented_cone_cluster1.png")
plot_point_cloud(feature_points["hull_vertices"], "Cone Feature Extraction - Edges", "illustrations/cone_edges.png")
