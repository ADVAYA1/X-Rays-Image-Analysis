# Custom CNN for Multi-Class Classification of Respiratory Diseases

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/ADVAYA1/X-Rays-Image-Analysis)


## Project Overview 📖

This project presents a robust deep learning solution for the automated diagnosis of respiratory diseases from chest X-ray images. A **custom Convolutional Neural Network (CNN)** was designed from the ground up to perform multi-class classification, accurately distinguishing between four categories: **COVID-19**, **Normal**, **Pneumonia**, and **Tuberculosis**. This tool is intended to assist medical professionals by providing rapid, high-accuracy diagnostics.

---

## Dataset & Preprocessing 📂

The model was trained on a comprehensive dataset of chest X-ray images, resized to `224x224` pixels. To enhance the model's ability to generalize and prevent overfitting, a key part of our methodology was **Image Augmentation**. The following transformations were applied to the training set:

* Rotation
* Width and Height Shifts
* Shear transformations
* Zoom
* Horizontal Flipping

---

## Custom CNN Architecture 🧠

A **custom CNN architecture** was developed using TensorFlow and Keras, tailored specifically for this classification task. The architecture was built sequentially, allowing for fine-grained control over the feature extraction and classification process.

* **Convolutional & Pooling Layers (`Conv2D`, `MaxPooling2D`):** Multiple layers were stacked to progressively extract hierarchical features from the X-ray images. `ReLU` activation was used to introduce non-linearity.
* **Dropout Layer:** A `Dropout` layer was strategically placed to mitigate overfitting by randomly deactivating neurons during training.
* **Flatten & Dense Layers:** The extracted feature maps were flattened into a 1D vector and passed through fully connected (`Dense`) layers. The final output layer utilizes a `softmax` activation function to generate class probabilities.

---

## Model Training & Evaluation 📊

The model's training process and performance were rigorously evaluated to ensure its effectiveness and reliability.

### Optimizer Performance Analysis

An essential step was to select the best optimization algorithm. A comparative analysis was conducted, and the **Adam optimizer** was chosen, as it achieved the highest validation accuracy and demonstrated superior convergence.

![Optimizer Accuracy Comparison](Optimisers%20Accuracy.png)

### Final Model Results

The final custom CNN, trained using the Adam optimizer, achieved high accuracy on the unseen test data. The performance is detailed in the **Confusion Matrix** and **Classification Report**, which confirm the model's strong predictive power, especially in identifying 'NORMAL' and 'TUBERCULOSIS' cases with high precision and recall.

![Confusion Matrix and Classification Report](Result.png)

---

## How to Run 🚀

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ADVAYA1/X-Rays-Image-Analysis.git](https://github.com/ADVAYA1/X-Rays-Image-Analysis.git)
    cd X-Rays-Image-Analysis
    ```
2.  **Prepare the Dataset:**
    Organize your dataset into a root folder with four sub-folders named `COVID19`, `NORMAL`, `PNEUMONIA`, and `TUBERCULOSIS`.
3.  **Run the Jupyter Notebook:**
    Open and execute the `MLFinal.ipynb` notebook. Ensure you update the directory paths to match the location of your dataset.

---

## Dependencies 📦

* TensorFlow & Keras
* Scikit-learn
* Matplotlib & Seaborn
* NumPy
