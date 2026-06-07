
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="/content/Engine_Maintenance_CICD/deployment",
    repo_id="Satyanjay/Engine-Predictive-Maintenance",
    repo_type="space"
)
