# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
#for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
#for creating folder
import os
#for hugging face soace authentication to upload
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_experiment")

api =  HfApi()


Xtrain_path = "hf://Satyanjay/engine-condition-monitoring-CICD/Xtrain.csv"
Xtest_path = "hf://Satyanjay/engine-condition-monitoring-CICD/Xtest.csv"
ytrain_path = "hf://Satyanjay/engine-condition-monitoring-CICD/ytrain.csv"
ytest_path = "hf://Satyanjay/engine-condition-monitoring-CICD/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

print("Data loaded successfully")

# Numeric features
numeric_features = [
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp'
]

#Set the class weight to handel class imbalance
class_weight =  ytrain.value_counts()[1] / ytrain.value_counts()[0]
class_weight

# Define the preprocessing steps
preprocessor =  make_column_transformer((standardScaler(), numeric_features))

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)  
rf_model = RandomForestClassifier(random_state=42)

xgb_param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
}

rf_param_grid = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [5, 10],
} 

# Pipelines
model_pipeline = make_pipeline(preprocessor, xgb_model)
model_pipeline = make_pipeline(preprocessor, rf_model)

#Start MLflow run
# Function to train, tune, and log
def train_and_log(model_name, pipeline, param_grid):
    with mlflow.start_run(run_name=model_name):
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_
        y_pred_test = best_model.predict(Xtest)

        report = classification_report(ytest, y_pred_test, output_dict=True)

        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({
            "test_accuracy": report['accuracy'],
            "test_precision": report['1']['precision'],
            "test_recall": report['1']['recall'],
            "test_f1-score": report['1']['f1-score']
        })

        # Log model
        mlflow.sklearn.log_model(best_model, model_name)

        return report['accuracy']

# Train both models
xgb_acc = train_and_log("XGBoost", xgb_pipeline, xgb_param_grid)
rf_acc = train_and_log("RandomForest", rf_pipeline, rf_param_grid)

# Compare
print("XGBoost Accuracy:", xgb_acc)
print("RandomForest Accuracy:", rf_acc)

best_model_name = "XGBoost" if xgb_acc > rf_acc else "RandomForest"
print("Best model is:", best_model_name)

# Save the model locally
model_path = "best_model.joblib"
joblib.dump(best_model, model_path)

# Log the model artifact
mlflow.log_artifact(model_path, artifact_path="model")
print(f"Model saved as artifact ar: {model_path}")
# Upload the model to Hugging Face
repo_id = "Satyanjay/engine-condition-monitoring-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it")  
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created successfully")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type
)               
