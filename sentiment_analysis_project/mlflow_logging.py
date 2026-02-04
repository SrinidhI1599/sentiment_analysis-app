# importing required libraries
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Set MLflow tracking server URI (where MLflow will log runs and artifacts)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set (or create) an experiment in MLflow to organize runs
mlflow.set_experiment("sentiment_analysis_experiment_1")

def log_model(model, model_name, X_test, y_test):
    """
    Logs a trained model and its evaluation metrics to MLflow.

    Parameters:
    - model: trained scikit-learn model
    - model_name: string name to identify the run/model
    - X_test: test feature data
    - y_test: true labels for test data
    """
    
    # Start a new MLflow run with the given model name
    with mlflow.start_run(run_name=model_name):

        # Generate predictions on the test set            
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")

        # Print metrics to console for quick inspection
        print(f"{model_name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Log metrics to MLflow for tracking and visualization
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)

        # Log the trained model to MLflow using Skops format (safe serialization for sklearn models)
        mlflow.sklearn.log_model(
            model,
            artifact_path=model_name,
            serialization_format="skops"
        )
