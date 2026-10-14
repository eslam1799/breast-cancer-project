
from production_app.data import load_data
from production_app.model_service import load_model


def main():
    # Load the saved model; do not train it
    model = load_model()

    # Get sample inputs for this demonstration
    X, y = load_data()
    X_sample = X.iloc[:5]

    # Predict using the saved model
    predictions = model.predict(X_sample)

    print("\nPredictions:", predictions)
    print("Actual labels:", y.iloc[:5].tolist())


if __name__ == "__main__":
    main()
