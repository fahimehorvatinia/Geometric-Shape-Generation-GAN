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

def plot_normals(points, normals, title, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='blue')
    # Plot the normals as arrows
    for i in range(0, len(points), 10):  # Plot fewer normals for clarity
        ax.quiver(points[i, 0], points[i, 1], points[i, 2],
                  normals[i, 0], normals[i, 1], normals[i, 2],
                  length=0.1, color='red')
    ax.set_title(title)
    plt.savefig(output_path)
    plt.close()

# Load your segmented points and features (adjust paths as needed)
segmented_points = np.load("3D_Shape_Dataset/Segmented/Cone_segmented.npy", allow_pickle=True).item()
feature_points = np.load("3D_Shape_Dataset/Features/Cone_features.npy", allow_pickle=True).item()

# Assuming 'hull_vertices' holds points and 'normals' holds normals
points = feature_points["hull_vertices"]
normals = feature_points["normals"]

# Generate the illustration
plot_normals(points, normals, "Surface Normals for Cone", "illustrations/cone_normals.png")