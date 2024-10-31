import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# List of shapes and their corresponding files
shapes = {
    'Cone': '3D_Shape_Dataset/Cone_points.npy',
    'Cube': '3D_Shape_Dataset/Cube_points.npy',
    'Cylinder': '3D_Shape_Dataset/Cylinder_points.npy',
    'Octahedron': '3D_Shape_Dataset/Octahedron_points.npy',
    'Pyramid': '3D_Shape_Dataset/Pyramid_points.npy',
    'Sphere': '3D_Shape_Dataset/Sphere_points.npy',
    'Star': '3D_Shape_Dataset/Star_points.npy',
    'Tetrahedron': '3D_Shape_Dataset/Tetrahedron_points.npy',
    'Torus': '3D_Shape_Dataset/Torus_points.npy'
}


# Function to plot and save 3 different angles of a shape
def plot_3d_shape(points_file, shape_name):
    points = np.load(points_file)

    for i, angle in enumerate([30, 60, 90]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.view_init(30, angle)  # Change the viewing angle for variety
        ax.set_title(f'{shape_name} - View {i + 1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Save each view as an image
        plt.savefig(f'{shape_name}_View_{i+1}.png')
        plt.close()


# Generate and save images for all shapes
for shape_name, points_file in shapes.items():
    plot_3d_shape(points_file, shape_name)
