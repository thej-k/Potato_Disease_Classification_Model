# Potato Disease Classification using TensorFlow

## Overview
This project implements a **Convolutional Neural Network (CNN)** model using **TensorFlow** to classify potato leaf diseases into three categories:
- **Potato_early_blight**
- **Potato_late_blight**
- **Potato_healthy**

This model aims to assist farmers and agricultural experts by identifying common potato diseases through image classification, ultimately reducing crop damage and improving yield.

## Dataset
The dataset used contains images of potato leaves, categorized into:
1. **Potato_early_blight**
2. **Potato_late_blight**
3. **Potato_healthy**

The dataset can be sourced from [Kaggle](https://www.kaggle.com), which contains labeled images for training, validation, and testing.

## Model Architecture
The model follows a **Convolutional Neural Network (CNN)** architecture with several convolutional layers for feature extraction and dense layers for classification. The key layers include:
- **Conv2D** layers for feature extraction.
- **MaxPooling2D** layers for dimensionality reduction.
- **Flatten** and **Dense** layers for classification.
- **Softmax** activation for multi-class classification.

### Model Summary
To view the model summary, run the following code:
```python
model.summary()
```

## Future Enhancements
- **Mobile App Deployment**: Deploy the model as a mobile application to allow farmers to upload images and receive real-time disease classification.
- **Transfer Learning**: Experiment with pre-trained models to improve classification accuracy and reduce training time.
- **Cloud Integration**: Deploy the model on cloud platforms (e.g., AWS, GCP) for real-time access and scalability.

## Technologies Used
- **TensorFlow** and **Keras** for deep learning model building.
- **Python** for scripting and data processing.
- **NumPy** for numerical computations.
- **Matplotlib** for data visualization.
