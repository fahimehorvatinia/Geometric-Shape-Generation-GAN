# 🎨 Parametric Geometric Shape Generation Using GANs

## Introduction

- 🏗️ **Goal**: Build a GAN that generates **parametrically encoded geometric shapes**.
- 🔺 **Shapes**: Polygons, circles, stars.
- 📝 **Output**: Coordinates of vertices, radii, or angular parameters instead of images.
- 🔄 **Process**: Convert parametric data into images using OpenCV or a similar library.
- 🌟 **Focus**: 
  - Generate valid shapes (e.g., no collinear points in polygons).
  - Ensure shapes maintain their geometric properties.

---

## Problem Definition

- 🎯 **Objective**: GAN generates **parametric representations** of geometric shapes.
- 🔢 **Inputs**:
  - Random noise.
  - Optional class labels (for Conditional GAN).
- 📊 **Outputs**:
  - **Polygons**: Coordinates of vertices.
  - **Circles**: Center point and radius.
  - **Stars**: Vertex coordinates with angular constraints.
- ⚠️ **Key challenge**: Ensure generated parameters correspond to **valid shapes**.

---

## Dataset Description

- 🛠️ **Synthetic Dataset**: Created using OpenCV and NumPy.
- 📐 **Shape Classes**: Polygons, circles, stars.
- 🗂️ **Data Structure**:
  - **Polygons**: Vertex coordinates.
  - **Circles**: Center and radius.
  - **Stars**: Vertex coordinates with angular constraints.
- 📊 **Data Splits**:
  - Training set: Parametric representations for model training.
  - Validation set: For tuning and avoiding overfitting.
  - Test set: For final evaluation of GAN performance.

---

## High-Level Solution

### GAN Architecture:
- **Generator**: The generator will take random noise as input and output parametric representations of shapes. For example, for a polygon, the output would be a set of coordinate pairs representing the vertices of the shape. The generator will be trained to produce valid sets of parameters that can be translated into meaningful shapes.
  
- **Discriminator**: The discriminator will take the generated parametric representation and validate whether it forms a valid geometric shape. For instance, for polygons, it will check that the vertices are not collinear and that the shape's structure adheres to its geometric properties.

### Translation to Images:
After generating the parametric representation, the shape will be drawn as an image using OpenCV or another visualization library. This process will allow for visual inspection of the generated shapes, but the GAN itself will operate directly on the parametric data.


---

## 📊 Evaluation Metrics

| **Metric**                | **What It Measures**                                                                                 | **How It’s Used**                                                                                         |
|---------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| **Geometric Correctness**      | Checks if shapes follow geometric rules (e.g., no collinear points in polygons, symmetry in circles)  | Evaluate if generated shapes adhere to geometric principles.                                                |
|  **Structural Similarity Index (SSIM)** | Measures perceptual similarity between generated and real shapes after translation to images    | Compare generated images to real shapes in terms of visual similarity.                                      |
|  **Coordinate Distance**        | Measures distance between vertices in generated and real polygons                                    | Ensure accuracy of generated polygon vertices.                                                              |
|  **Radius Deviation**           | Compares generated and real circle radii                                                             | Evaluate accuracy of generated circle parameters.                                                           |
|  **Fréchet Inception Distance (FID)** | Measures the distance between real and generated parametric data distributions                    | Compare parametric data distributions to assess shape diversity and quality.                                |
|  **Inception Score (IS)**       | Evaluates quality and diversity of parametric representations                                        | Ensure variety and correctness in generated shape categories.                                               |

---

## 🧠 What I Need to Learn

- **GAN for parametric data**:
  - How to generate parametric data instead of raw images.
- **Geometric validation**:
  - Implement checks to validate geometric properties (e.g., no collinear points, symmetry).
- **Evaluation metrics**:
  - Learn appropriate metrics for evaluating parametric data and their corresponding shapes.

---

## 🌟 Conclusion

- **Focus**: GAN generates parametric representations of geometric shapes, rather than images.
- **Advantages**:
  - Direct control over geometric properties.
  - Deeper understanding of parametric data generation.
- **Goal**: Train GAN to produce valid geometric shapes through parametric encoding.
