1. Project Goal

The goal of this project is to develop a machine learning pipeline for classifying breast cancer tumors as malignant or benign using Python and Scikit-learn.

The project demonstrates how to organize a machine learning application into reusable modules, train and evaluate a model, serialize the trained pipeline, and load the saved model for future predictions without retraining.

This is an educational project intended to demonstrate machine learning development and production-oriented code organization. It is not intended for clinical diagnosis or medical decision-making.

2. Dataset

This project uses the Wisconsin Breast Cancer Diagnostic dataset provided by Scikit-learn through sklearn.datasets.load_breast_cancer.

The dataset contains 569 samples and 30 numerical features describing characteristics of cell nuclei derived from breast mass images.

The target labels are:

0: Malignant

1: Benign

The dataset is loaded directly through Scikit-learn, so a separate dataset download is not required.

The data is divided into training and testing sets using an 80/20 split. Stratification is applied to preserve the target-class proportions, and random_state=42 makes the split reproducible.

3. Machine Learning Pipeline

The project follows these steps:

Data loading: Load the breast cancer dataset into feature and target variables.

Data splitting: Divide the data into training and testing sets.

Preprocessing: Apply StandardScaler to standardize feature values.

Model training: Train a Logistic Regression classifier.

Model evaluation: Measure performance using accuracy, malignant-class precision, recall, F1-score, a confusion matrix, and a classification report.

Model serialization: Save the complete preprocessing and classification pipeline using Joblib.

Saved-model prediction: Load the serialized pipeline and use it to generate predictions without retraining.

The scaler and classifier are combined in a Scikit-learn Pipeline. This ensures that the same fitted preprocessing transformation is applied when the saved model makes predictions.

4. Model Choice

The selected algorithm is Logistic Regression.

It was chosen because it is a widely used classification algorithm, provides a straightforward baseline, and is relatively easy to interpret compared with more complex models.

Because the features have different numerical scales, StandardScaler is applied before Logistic Regression.
