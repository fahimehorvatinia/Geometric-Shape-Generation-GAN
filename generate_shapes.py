import numpy as np
import os

# Function to save point cloud to a file
def save_point_cloud(points, shape_name, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, f"{shape_name}_points.npy")
    np.save(file_path, points)

# Functions to generate the different 3D shapes
def generate_sphere_points(radius, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z]).T

def generate_tetrahedron_points(scale, num_points):
    vertices = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]) * scale
    points = np.random.uniform(0, 1, (num_points, 4))
    points /= np.sum(points, axis=1)[:, None]
    random_points = np.dot(points, vertices)
    return random_points

def generate_cube_points(scale, num_points):
    return np.random.uniform(-scale, scale, (num_points, 3))

def generate_cylinder_points(radius, height, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(-height / 2, height / 2, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.array([x, y, z]).T

def generate_cone_points(radius, height, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.random.uniform(0, radius, num_points)
    z = np.random.uniform(0, height, num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y, z]).T

def generate_star_points(scale, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points)
    r = np.random.uniform(0.5 * scale, scale, num_points)
    x = r * np.sin(angles)
    y = r * np.cos(angles)
    z = np.random.uniform(-scale, scale, num_points)
    return np.array([x, y, z]).T

def generate_torus_points(inner_radius, outer_radius, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    r = (outer_radius - inner_radius) / 2
    x = (inner_radius + r * np.cos(phi)) * np.cos(theta)
    y = (inner_radius + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.array([x, y, z]).T

def generate_pyramid_points(base_size, height, num_points):
    vertices = np.array([[0, 0, height], [base_size, base_size, 0], [-base_size, base_size, 0],
                         [-base_size, -base_size, 0], [base_size, -base_size, 0]])
    points = np.random.uniform(0, 1, (num_points, 5))
    points /= np.sum(points, axis=1)[:, None]
    random_points = np.dot(points, vertices)
    return random_points

def generate_octahedron_points(scale, num_points):
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) * scale
    points = np.random.uniform(0, 1, (num_points, 6))
    points /= np.sum(points, axis=1)[:, None]
    random_points = np.dot(points, vertices)
    return random_points

# Generate dataset folder
output_folder = "3D_Shape_Dataset"

# Define shapes and generate datasets
shapes = {
    "Sphere": generate_sphere_points(1, 1000),
    "Tetrahedron": generate_tetrahedron_points(1, 1000),
    "Cube": generate_cube_points(1, 1000),
    "Cylinder": generate_cylinder_points(1, 2, 1000),
    "Cone": generate_cone_points(1, 2, 1000),
    "Star": generate_star_points(1, 1000),
    "Torus": generate_torus_points(1, 2, 1000),
    "Pyramid": generate_pyramid_points(1, 2, 1000),
    "Octahedron": generate_octahedron_points(1, 1000),
}

# Save point clouds for each shape
for shape_name, points in shapes.items():
    save_point_cloud(points, shape_name, output_folder)

print(f"3D shape dataset saved in folder '{output_folder}'")
