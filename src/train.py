import pandas as pd
import joblib
import mlflow
from mlflow import sklearn as mlflow_sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')
from mlflow.models import infer_signature

# 1. Setup MLflow Experiment
mlflow.set_experiment("Delhi_Traffic_ETA_Predictor")

def run_mlops_pipeline():
    print("🚀 Booting Elite Training Pipeline...")
    
    # 2. Ingest the Cleaned Data from the Notebook
    data_path = "data/processed/master_training_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}. Run notebooks/data_exploration.ipynb first!")
    
    df = pd.read_csv(data_path)
    
    # 3. Dynamic Feature Selection
    target_col = 'travel_time_minutes'
    
    # Exclude IDs or any non-numeric columns that might have slipped through
    exclude_cols = ['Trip_ID', 'timestamp', 'Time', target_col] 
    features = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"🔍 Training on {len(features)} fused features...")
    
    X = df[features]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Define Hyperparameters (Slightly deeper because of the Uber data fusion)
    params = {
        "n_estimators": 200,
        "max_depth": 18,
        "min_samples_split": 4,
        "random_state": 42
    }

    # 5. Start MLflow Run
    with mlflow.start_run(run_name="RF_Delhi_Fused_Production"):
        # Log your parameters
        mlflow.log_params(params)
        
        print("🧠 Training Domain-Adapted Random Forest Model...")
        print("🧠 Training Domain-Adapted Random Forest Model...")
        # Explicitly passing parameters silences Pylance's type-checker panic
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["random_state"]
        )
        model.fit(X_train, y_train)
        
        print("📊 Evaluating Model Performance...")
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        
        # Log your metrics
        mlflow.log_metric("MAE_mins", mae)
        mlflow.log_metric("R2_score", r2)
        mlflow.log_metric("RMSE_mins", rmse)
        
        print(f"🏆 Results -> MAE: {mae:.2f} mins | R2: {r2:.4f}")
        
        # 6. Save the production .joblib files for the FastAPI backend
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/vanet_rf_model.joblib")
        joblib.dump(features, "models/vanet_model_columns.joblib")
        
        # --- THE NEW UI VISIBILITY UPGRADE ---
        
        # A. Create a human-readable text file of the features
        with open("models/feature_schema.txt", "w") as f:
            f.write("FUSED MODEL FEATURE SCHEMA\n")
            f.write("="*30 + "\n")
            for idx, col in enumerate(features):
                f.write(f"{idx + 1}. {col}\n")
                
        # B. Infer the exact mathematical signature (Input types -> Output types)
        signature = infer_signature(X_train, predictions)
        
        # 7. Log everything to MLflow
        mlflow.log_artifact("models/vanet_rf_model.joblib")
        mlflow.log_artifact("models/vanet_model_columns.joblib")
        mlflow.log_artifact("models/feature_schema.txt") # Logs the readable text
        
        # 8. Log the model natively with its Signature
        mlflow_sklearn.log_model(
            sk_model=model,
            artifact_path="fused_random_forest",
            signature=signature
        )
        
        print("✅ Pipeline Complete. Features and Signatures logged to MLflow UI!")

if __name__ == "__main__":
    # Ensure we run from the project root
    if not os.path.exists("data"):
        print("⚠️ Please run this script from the root 'traffic_predictor_v2' directory!")
    else:
        run_mlops_pipeline()