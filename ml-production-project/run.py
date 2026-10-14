
import logging

from production_app.logging_config import setup_logging
from production_app.data import load_data, split_data
from production_app.training import train_model
from production_app.evaluation import evaluate_model
from production_app.model_service import save_model, load_model


def main():
    # 1. Configure logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Application started")

    # 2. Load dataset
    logger.info("Loading dataset")
    X, y = load_data()
    logger.info("Dataset loaded successfully")

    # 3. Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info("Data split completed")

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 4. Train model
    logger.info("Starting model training")
    model = train_model(X_train, y_train)
    logger.info("Model training completed")

    # 5. Evaluate model
    logger.info("Starting model evaluation")
    evaluate_model(model, X_test, y_test)
    logger.info("Model evaluation completed")

    # 6. Save trained model
    logger.info("Saving trained model")
    save_model(model)

    # 7. Load saved model to verify serialization
    logger.info("Loading saved model")
    saved_model = load_model()

    # 8. Test predictions using the saved model
    predictions = saved_model.predict(X_test.head(5))

    print("\nPredictions:", predictions)
    print("Actual labels:", y_test.head(5).to_numpy())

    logger.info("Prediction test completed")
    logger.info("Application finished")


if __name__ == "__main__":
    main()
