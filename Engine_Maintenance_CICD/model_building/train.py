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
class_weight = (ytrain == 1).sum() / (ytrain == 0).sum()

# ✅ preprocessing fix
preprocessor = make_column_transformer((StandardScaler(), numeric_features), remainder ='passthrough')

# Models
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric='logloss')


# Best Parameters
xgb_param = {
    'xgbclassifier__n_estimators': 75,
    'xgbclassifier__max_depth': 4,
    'xgbclassifier__colsample_bylevel': 0.6,
    'xgbclassifier__learning_rate': 0.1,
    'xgbclassifier__reg_lambda': 0.4,
}


# Fitting best mdoel

best_model = make_pipeline(preprocessor, xgb_model)
best_model.set_params(**xgb_param)
best_model.fit(Xtrain, ytrain)

# Model evaluation

y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

train_report = classification_report(ytrain, y_pred_train, output_dict=True)
test_report = classification_report(ytest, y_pred_test, output_dict=True)

print("Train Accuracy:", train_report['accuracy'])
print("Test Accuracy:", test_report['accuracy'])
print("Train Precision:", train_report['1']['precision'])
print("Test Precision:", test_report['1']['precision'])
print("Train Recall:", train_report['1']['recall'])
print("Test Recall:", test_report['1']['recall'])


# Saving the best model in hugging face space


# Save model
model_path = "best_model.joblib"
joblib.dump(best_model, model_path)


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

