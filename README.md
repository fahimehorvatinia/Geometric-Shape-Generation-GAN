# Geometric Shape Generation Using GANs

## 1. Introduction

In this project, I aim to explore the capabilities of Generative Adversarial Networks (GANs) by generating 2D geometric shapes, such as circles, triangles, squares, and polygons. This problem is particularly interesting because it offers a simplified, noise-free dataset where the focus can be placed on understanding the generative process without the complexities found in real-world data. The synthetic nature of the data will allow me to control important parameters, such as shape size, rotation, and color, which can help isolate learning outcomes for the GAN.

---

## 2. Problem Definition

The primary goal of this project is to train a GAN model capable of generating 2D geometric shapes that resemble a set of predefined shapes such as circles, squares, triangles, and polygons. The generator will learn to produce synthetic shapes from random noise, while the discriminator will evaluate whether the shapes generated are geometrically correct. A key challenge lies in ensuring that the GAN can capture essential geometric properties like edges and symmetry in the generated shapes.

---

## 3. Dataset Description

The dataset used in this project is composed of 9 distinct geometric shapes: triangles, squares, pentagons, hexagons, heptagons, octagons, nonagons, circles, and stars. Each shape is randomly generated on 200x200 RGB images, with varying perimeters, positions, and rotations. Additionally, both the background and fill colors of each shape are randomly selected. The dataset contains 10,000 images per shape class, for a total of 90,000 images.

To ensure proper evaluation of the GAN’s performance, the dataset will be divided into three subsets:

- **Training Set**: 70% of the data (63,000 images) will be used for model training.
- **Validation Set**: 15% (13,500 images) will be used to fine-tune the model and prevent overfitting.
- **Test Set**: 15% (13,500 images) will be reserved for final model evaluation.

---

## 4. High-Level Solution Overview

The proposed solution will employ a GAN consisting of two main components: a generator and a discriminator. The generator will take random noise as input and produce 200x200 RGB images of geometric shapes, while the discriminator will evaluate whether the generated images are real or synthetic. To guide the generator in producing specific shapes, a Conditional GAN (cGAN) architecture may be employed, where the generator and discriminator will both be conditioned on the shape class (e.g., circle, square, etc.).

The project will begin with a simple GAN implementation, and later extend to more complex architectures (such as DCGAN or cGAN) if necessary to improve performance.

---

## 5. What I Need to Learn

To successfully implement this project, I will need to deepen my understanding of the following:

- **GAN architecture**: I am familiar with the basic structure of GANs, but I need to study how to fine-tune the generator and discriminator to handle geometric shapes effectively.
- **Training stability**: Training the GAN is difficult. I need to explore methods for stabilizing GAN training and avoiding issues such as mode collapse.
- **Evaluation metrics**: I will need to study evaluation methods such as the Structural Similarity Index (SSI) and Mean Square Error (MSE) and how to compare generated shapes with the ground truth shapes in a meaningful way.

---

## 6. Conclusion

In conclusion, this project will explore how GANs can be used to generate synthetic images of 2D geometric shapes. By leveraging a dataset of 9 different shapes and focusing on learning the fundamentals of GAN architecture, I aim to gain deeper insights into the challenges and capabilities of generative modeling. The ultimate goal is to train a GAN capable of producing realistic geometric shapes while mastering key techniques in generative AI.
