# Breast Cancer Classification — ML Production Project

## Overview

This project demonstrates a machine learning workflow for breast cancer classification using Python and Scikit-learn. It covers data loading, preprocessing, model training, evaluation, and model serialization.

The project uses the Wisconsin Breast Cancer dataset available through Scikit-learn.

## Objectives

- Explore and analyze the dataset using Jupyter Notebook.
- Prepare and split data into training and testing sets.
- Train a Logistic Regression classifier using a Scikit-learn pipeline.
- Evaluate model performance using classification metrics.
- Save and reload the trained pipeline using Joblib.
- Organize reusable ML code into separate Python modules.

## Project Structure

```text
ml-production-project/
├── notebooks/
│   └── breast_cancer_analysis.ipynb
├── production_app/
│   ├── __init__.py
│   ├── data.py
│   ├── training.py
│   ├── evaluation.py
│   └── model_service.py
├── models/
│   └── .gitkeep
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
├── run.py
└── predict.py
```

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook
- Joblib
- uv for Python project and dependency management

## Setup

Install Python 3.12 or another version supported by `pyproject.toml`, and install uv.

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/YOUR_USERNAME/ml-production-project.git
cd ml-production-project
```

Install the project dependencies:

```bash
uv sync
```

## Run the Project

Train and evaluate the model, then save it:

```bash
uv run python run.py
```

Load the saved model and generate predictions:

```bash
uv run python predict.py
```

## Notebook

The `notebooks/` directory contains the exploratory data analysis, visualizations, and feature-analysis work.

## Model Evaluation

The evaluation module reports classification metrics and a confusion matrix. The target labels in the Scikit-learn dataset are `0` for malignant and `1` for benign.

## Limitations

This is an educational machine learning project. Its predictions are not intended for clinical diagnosis or medical decision-making.

## Author

Add your name and any relevant course or project information here.