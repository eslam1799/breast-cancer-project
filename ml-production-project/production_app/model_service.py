
from pathlib import Path

import joblib


# Find the models folder from this Python file's location
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "breast_cancer_pipeline.joblib"
)


def save_model(model):
    """Save the trained model to a file."""

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


def load_model():
    """Load a previously saved model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No saved model found at {MODEL_PATH}. "
            "Train and save the model first."
        )

    model = joblib.load(MODEL_PATH)

    print("Saved model loaded successfully!")

    return model
