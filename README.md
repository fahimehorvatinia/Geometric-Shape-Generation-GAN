
# 🎨 Parametric Geometric Shape Generation Using GANs

## Introduction
In this project, I aim to explore how a Generative Adversarial Network (GAN) can be used to generate geometric shapes by focusing on their underlying parametric structure.  
🏗️ **Goal**: Build a GAN that generates **parametrically encoded geometric shapes**.  
🔺 **Shapes**: Polygons, circles, stars.  
📝 **Output**: Coordinates of vertices, radii, or angular parameters instead of images.  
🔄 **Process**: Convert parametric data into images using OpenCV or a similar library.  
🌟 **Focus**:  
  Generate valid shapes (e.g., no collinear points in polygons).  
  Ensure shapes maintain their geometric properties.

---

## Problem Definition

🎯 **Objective**: GAN generates **parametric representations** of geometric shapes.  
🔢 **Inputs**:  
  Random noise.  
  Optional class labels (for Conditional GAN).  
📊 **Outputs**:  
  **Polygons**: Coordinates of vertices.  
  **Circles**: Center point and radius.  
  **Stars**: Vertex coordinates with angular constraints.  
⚠️ **Key challenge**: Ensure generated parameters correspond to **valid shapes**.

---

## Dataset Description

🛠️ **Synthetic Dataset**: Created using OpenCV and NumPy.  
📐 **Shape Classes**: Polygons, circles, stars.  
🗂️ **Data Structure**:  
  **Polygons**: Vertex coordinates.  
  **Circles**: Center and radius.  
  **Stars**: Vertex coordinates with angular constraints.  
📊 **Data Splits**:  
  Training set: Parametric representations for model training.  
  Validation set: For tuning and avoiding overfitting.  
  Test set: For final evaluation of GAN performance.

---

## High-Level Solution

### GAN Architecture:
**Generator**: The generator will take random noise as input and output parametric representations of shapes. For example, for a polygon, the output would be a set of coordinate pairs representing the vertices of the shape. The generator will be trained to produce valid sets of parameters that can be translated into meaningful shapes.  

**Discriminator**: The discriminator will take the generated parametric representation and validate whether it forms a valid geometric shape. For instance, for polygons, it will check that the vertices are not collinear and that the shape's structure adheres to its geometric properties.

### Translation to Images:
After generating the parametric representation, the shape will be drawn as an image using OpenCV or another visualization library. This process will allow for visual inspection of the generated shapes, but the GAN itself will operate directly on the parametric data.

---

## 📊 Evaluation Metrics

| **Metric**                | **What It Measures**                                                                                 | **How It’s Used**                                                                                         |
|---------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| **Geometric Correctness**      | Checks if shapes follow geometric rules (e.g., no collinear points in polygons, symmetry in circles)  | Evaluate if generated shapes adhere to geometric principles.                                                |
| **Structural Similarity Index (SSIM)** | Measures perceptual similarity between generated and real shapes after translation to images    | Compare generated images to real shapes in terms of visual similarity.                                      |
| **Coordinate Distance**        | Measures distance between vertices in generated and real polygons                                    | Ensure accuracy of generated polygon vertices.                                                              |
| **Radius Deviation**           | Compares generated and real circle radii                                                             | Evaluate accuracy of generated circle parameters.                                                           |
| **Fréchet Inception Distance (FID)** | Measures the distance between real and generated parametric data distributions                    | Compare parametric data distributions to assess shape diversity and quality.                                |
| **Inception Score (IS)**       | Evaluates quality and diversity of parametric representations                                        | Ensure variety and correctness in generated shape categories.                                               |

---

## Part 2: Data Acquisition and Preparation

#### 1. Source of Data
The dataset was generated using a custom Python script with NumPy to create 3D parametric shapes. The shapes include Sphere, Tetrahedron, Cube, Cylinder, Cone, Star, Torus, Pyramid, and Octahedron. Each shape is represented as a 3D point cloud.

- **Number of Samples**: 1000 point clouds per shape
- **Total Samples**: 9000 point clouds

#### 2. Differences Between Train and Validation Subsets
- **Training Set**: Contains 60% (5400 point clouds) of the data.
- **Validation Set**: Contains 20% (1800 point clouds) of the data, used for validating the model's performance.
- **Test Set**: 20% (1800 point clouds) reserved for final evaluation.

#### 3. Number of Distinct Objects/Subjects
Each shape (e.g., Sphere, Tetrahedron) has 1000 samples, all generated with randomized parameters to ensure variability in each shape.
## 3D Shape Visualizations

### Cone
![Cone](dataset-examples/Cone_View_1.png)

### Cube
![Cube](dataset-examples/Cube_View_1.png)

### Cylinder
![Cylinder](dataset-examples/Cylinder_View_1.png)

### Octahedron
![Octahedron](dataset-examples/Octahedron_View_1.png)

### Pyramid
![Pyramid](dataset-examples/Pyramid_View_1.png)

### Sphere
![Sphere](dataset-examples/Sphere_View_1.png)

### Star
![Star](dataset-examples/Star_View_1.png)

### Tetrahedron
![Tetrahedron](dataset-examples/Tetrahedron_View_1.png)

### Torus
![Torus](dataset-examples/Torus_View_1.png)

#### 4. Characterization of Samples
- **Resolution**: 3D point clouds with 1000 points per shape.
- **Sensors Used**: N/A (synthetic data generated with Python).
- **Illumination/Conditions**: N/A (synthetic data).

---
## Part 3: Data Preprocessing, Segmentation, and Feature Extraction

### Overview
This section describes the data preprocessing, segmentation, and feature extraction methods applied to prepare 3D geometric shapes for GAN training in the "Geometric-Shape-Generation-GAN" project. The aim is to provide high-quality, structured input data for generating realistic parametric representations of shapes.

### 1. Data Preprocessing

#### Methods Applied
- **Normalization**: Each 3D point cloud is scaled to fit within a standard range of -1 to 1.
- **Noise Reduction**: Gaussian filtering is applied to reduce noise, which enhances the clarity of each shape and minimizes artifacts.

#### Justification
- **Normalization**: Essential for GAN stability, this ensures that input shapes have consistent scales, reducing issues with size variation during training.
- **Noise Reduction**: Reduces artifacts that could distort the GAN’s understanding of boundaries, providing cleaner data for training.

#### Example Illustration
**Original vs. Preprocessed Shape**  
![Original Shape](illustrations/original_cone.png)  
![Preprocessed Shape](illustrations/preprocessed_cone.png)

---

### 2. Segmentation

#### Methods Applied
- **Convex Hull Segmentation for Polygonal Shapes**: Convex Hull is used to extract boundary edges for shapes like tetrahedrons and cubes.
- **Surface Segmentation for Curved Shapes**: K-means clustering is applied to divide curved shapes (e.g., spheres, cones) into distinct regions.

#### Justification
- **Convex Hull**: By identifying boundary points, Convex Hull segmentation helps the GAN learn essential geometric constraints, such as edge boundaries and vertices.
- **K-means Clustering**: Enables clustering of curved surfaces, helping the GAN capture distinct surface regions and structural complexity.

#### Example Illustration
**Segmented Shape Example**  
![Convex Hull for Polygon](illustrations/segmented_cone_cluster1.png)

---

### 3. Feature Extraction

#### Methods Applied
- **Edge Detection and Surface Normals**:
  - **Edge Detection** for polygonal shapes: Uses Convex Hull to identify and emphasize edges.
  - **Surface Normals** for curved shapes: Calculates normals for each region, helping the GAN understand the 3D orientation.

#### Justification
- **Edge Detection**: Ensures that generated polygons respect geometric properties, such as edges and vertices, which are crucial for structural accuracy.
- **Surface Normals**: For realistic representations, normals guide the GAN in producing surfaces with the correct orientation.

#### Example Illustration
**Feature Extraction - Edges and Normals**  
![Edges](illustrations/cone_edges.png)  
![Normals](illustrations/cone_normals.png)

---

### Running the Code

Each step of the process is encapsulated in separate Python scripts. To execute each step, follow the commands below.

1. **Data Preprocessing**:
    ```bash
    python preprocessing.py
    ```
    This will create normalized and noise-reduced point clouds in `3D_Shape_Dataset/Preprocessed/`.

2. **Segmentation**:
    ```bash
    python segmentation.py
    ```
    This will generate segmented data in `3D_Shape_Dataset/Segmented/`, using Convex Hull for polygons and K-means clustering for curved shapes.

3. **Feature Extraction**:
    ```bash
    python feature_extraction.py
    ```
    Extracted features, including edges and normals, are saved in `3D_Shape_Dataset/Features/`.


--- 


## 🌟 Conclusion
This project seeks to show how GANs can learn the rules behind geometric shapes, giving us a deeper understanding of how to create meaningful forms, not just images.  
**Focus**: GAN generates parametric representations of geometric shapes, rather than images.
