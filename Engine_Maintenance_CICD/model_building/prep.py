
# -------------------------
# Import libraries
# -------------------------
import pandas as pd
import os

from sklearn.model_selection import train_test_split


# Hugging Face API
from huggingface_hub import HfApi

# -------------------------
# Load dataset
# -------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/Satyanjay/engine-condition-monitoring-CICD/engine_data.csv"
df = pd.read_csv(DATASET_PATH)

print(" Dataset loaded successfully")
print(" Columns:", df.columns)

# -------------------------
# Fix column name issue
# -------------------------
df.rename(columns={"lub oil temp": "Lub oil temp"}, inplace=True)

# -------------------------
# Handle missing values
# -------------------------
df.dropna(inplace=True)

# -------------------------
# Check imbalance
# -------------------------
print("\n Original Class Distribution:")
print(df["Engine Condition"].value_counts())

# -------------------------
# Define target
# -------------------------
target_col = "Engine Condition"

# -------------------------
# Split features and target
# -------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------
# Train-test split (stratified)
# -------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # maintain class ratio
)

print("\n Train-Test split completed")

# -------------------------
# Save datasets
# -------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("\n Files saved locally")

# -------------------------
# Upload to Hugging Face
# -------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

repo_id = "Satyanjay/engine-condition-monitoring-CICD"

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_id,
        repo_type="dataset",
    )

print("\n Data uploaded successfully")
