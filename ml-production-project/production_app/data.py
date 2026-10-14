
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from production_app.config import TEST_SIZE, RANDOM_STATE


def load_data():
    """Load the breast cancer dataset."""

    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target

    return X, y


def split_data(X, y):
    """Split data into training and testing sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
