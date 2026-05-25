# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score

# for model serialization
import joblib

# for creating folder
import os

# for hugging face authentication
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_experiment")

api = HfApi()
login(token=os.getenv("HF_TOKEN"))

# ✅ correct HF paths
Xtrain_path = "hf://datasets/Satyanjay/engine-condition-monitoring-CICD/Xtrain.csv"
Xtest_path = "hf://datasets/Satyanjay/engine-condition-monitoring-CICD/Xtest.csv"
ytrain_path = "hf://datasets/Satyanjay/engine-condition-monitoring-CICD/ytrain.csv"
ytest_path = "hf://datasets/Satyanjay/engine-condition-monitoring-CICD/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

print("Data loaded successfully")

# ✅ convert target properly
ytrain = ytrain.values.ravel()
ytest = ytest.values.ravel()

# Numeric features
numeric_features = [
    'Engine rpm',
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'Lub oil temp',
    'Coolant temp'
]

# ✅ class weight fix
class_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

# ✅ preprocessing fix
preprocessor = make_column_transformer((StandardScaler(), numeric_features))

# Models
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)  
rf_model = RandomForestClassifier(random_state=42)

xgb_param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

rf_param_grid = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [5, 10],
    'randomforestclassifier__min_samples_split': [2, 5],
    'randomforestclassifier__min_samples_leaf': [1, 2],
    'randomforestclassifier__max_features': ['sqrt', 'log2'],       
} 

# ✅ FIXED pipelines
xgb_pipeline = make_pipeline(preprocessor, xgb_model)
rf_pipeline = make_pipeline(preprocessor, rf_model)

# Training function
def train_and_log(model_name, pipeline, param_grid):
    with mlflow.start_run(run_name=model_name):
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_
        y_pred_test = best_model.predict(Xtest)

        report = classification_report(ytest, y_pred_test, output_dict=True)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({
            "test_accuracy": report['accuracy'],
            "test_precision": report['1']['precision'],
            "test_recall": report['1']['recall'],
            "test_f1-score": report['1']['f1-score']
        })

        mlflow.sklearn.log_model(best_model, model_name)

        # ✅ return both
        return best_model, report['accuracy']

# ✅ FIXED unpacking
xgb_model_obj, xgb_acc = train_and_log("XGBoost", xgb_pipeline, xgb_param_grid)
rf_model_obj, rf_acc = train_and_log("RandomForest", rf_pipeline, rf_param_grid)

print("XGBoost Accuracy:", xgb_acc)
print("RandomForest Accuracy:", rf_acc)

# ✅ FIXED comparison
best_model_name = "XGBoost" if xgb_acc > rf_acc else "RandomForest"
print("Best model is:", best_model_name)

# ✅ FIXED best model selection
best_model = xgb_model_obj if xgb_acc > rf_acc else rf_model_obj

# Save model
model_path = "best_model.joblib"
joblib.dump(best_model, model_path)

# Log model
mlflow.log_artifact(model_path, artifact_path="model")
print(f"Model saved as artifact at: {model_path}")

# Upload model
repo_id = "Satyanjay/engine-condition-monitoring-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type
)

print("✅ Model uploaded successfully")
