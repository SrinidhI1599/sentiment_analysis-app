# importing required libraries
from preprocessing import load_data, perform_eda, preprocess_data
from train_models import train_models
from mlflow_logging import log_model

# 1. Load and Explore Data
data_path = r"C:\Users\ragha\python_files\innomatics\sentiment_analysis_datasets\reviews_badminton\data.csv"
df = load_data(data_path)

# Perform EDA 
perform_eda(df)

# 2. Preprocess Data
X_train, X_test, y_train, y_test = preprocess_data(df)


# 3. Train Models
models = train_models(X_train, y_train)

# 4. Log Models with MLflow
for name, model in models.items():
    log_model(model, name, X_test, y_test)
