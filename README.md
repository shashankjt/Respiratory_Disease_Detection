# Lung Disease Classification using Hybrid DenseNet and ResNet

This project implements a deep learning model for classifying lung diseases from chest X-ray images. The model utilizes a hybrid architecture combining DenseNet121 and ResNet50 to leverage the strengths of both networks for improved feature extraction and classification performance.

## Dataset

The dataset used in this project is the "Lungs Disease Dataset 4 Types" available on Kaggle ([https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types](https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types)). It contains images categorized into five classes:

*   Bacterial Pneumonia
*   Corona Virus Disease
*   Normal
*   Tuberculosis
*   Viral Pneumonia

## Model Architecture

The model is a hybrid convolutional neural network (CNN) that combines pre-trained DenseNet121 and ResNet50 models. The features extracted by both networks are concatenated and fed into dense layers for classification.

## Project Structure

The notebook demonstrates the following steps:

1.  **Data Download:** Downloading the dataset from Kaggle using `kagglehub`.
2.  **Data Loading and Augmentation:** Loading the images using `ImageDataGenerator` with various augmentation techniques to improve model robustness.
3.  **Model Building:** Constructing the hybrid DenseNet + ResNet model.
4.  **Model Compilation:** Compiling the model with the Adam optimizer and categorical crossentropy loss.
5.  **Model Training:** Training the model with early stopping to prevent overfitting.
6.  **Model Saving:** Saving the trained model.
7.  **Evaluation:** Evaluating the model's performance on the test set using:
    *   Test Loss and Accuracy
    *   Confusion Matrix
    *   Classification Report (Precision, Recall, F1-score)
    *   ROC Curve (One-vs-Rest)
8.  **Prediction:** Demonstrating how to make predictions on a new image.

## Results

The model achieved the following performance on the test dataset:

*   **Test Loss:** 0.4242
*   **Test Accuracy:** 0.8765

The confusion matrix and classification report provide detailed insights into the model's performance for each class.

## Getting Started

### Prerequisites

*   Python 3.6+
*   TensorFlow
*   Keras
*   Numpy
*   Matplotlib
*   Seaborn
*   Pandas
*   Scikit-learn
*   Kagglehub


