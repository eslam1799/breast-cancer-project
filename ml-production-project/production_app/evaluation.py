import logging
from production_app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

logger.info("Prediction application started")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model using test data."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Class 0 represents malignant
    malignant_precision = precision_score(
        y_test, predictions, pos_label=0, zero_division=0
    )

    malignant_recall = recall_score(
        y_test, predictions, pos_label=0, zero_division=0
    )

    malignant_f1 = f1_score(
        y_test, predictions, pos_label=0, zero_division=0
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Malignant precision: {malignant_precision:.4f}")
    print(f"Malignant recall: {malignant_recall:.4f}")
    print(f"Malignant F1-score: {malignant_f1:.4f}")

    print("\nConfusion Matrix (rows = actual, columns = predicted)")
    print("Labels: 0 = malignant, 1 = benign")
    print(matrix)

    print("\nClassification Report")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["malignant", "benign"],
        zero_division=0,
    ))

    return {
        "accuracy": accuracy,
        "malignant_precision": malignant_precision,
        "malignant_recall": malignant_recall,
        "malignant_f1": malignant_f1,
        "confusion_matrix": matrix,
    }
