

from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Updated repo details for your project
repo_id = "Satyanjay/engine-condition-monitoring-CICD"
repo_type = "dataset"

# 🔹 Get token
token = os.getenv("HF_TOKEN")

if token is None:
    raise ValueError("HF_TOKEN is missing! Add it to GitHub Secrets.")

# 🔹 Initialize API
api = HfApi(token=token)

# 🔹 Check if dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f" Dataset '{repo_id}' already exists.")

except RepositoryNotFoundError:
    print(f" Dataset '{repo_id}' not found. Creating...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=token
    )

    print(f" Dataset '{repo_id}' created successfully.")

# 🔹 Upload data folder (UPDATED PATH)
data_folder_path = os.path.join(os.getcwd(), "Engine_Maintenance_CICD", "data")

if not os.path.exists(data_folder_path):
    raise FileNotFoundError(f"Data folder not found at: {data_folder_path}")

api.upload_folder(
    folder_path=data_folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Data uploaded successfully.")
