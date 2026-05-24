from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo, upload_file
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Register Dataset Repository
# -----------------------------
repo_id = "Satyanjay/tourism-package-prediction-CICD"
repo_type = "dataset"

# Create master folder and subfolder 'data'
os.makedirs("/content/tourism_project_CICD/data", exist_ok=True)

# Initialize Hugging Face API
api = HfApi()

try:
    # Try to access the repo
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    # If repo not found, create it
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    print(f"Repository '{repo_id}' created successfully.")

print("Data folder structure ready and repository registered.")

# -----------------------------
# Load Dataset for EDA
# -----------------------------
file_path = "/content/Engine_Maintenance_CICD/data/engine_data.csv"
df = pd.read_csv(file_path)

print("\n First 5 rows:")
print(df.head())

print("\n Shape of dataset:", df.shape)
print("\n Summary statistics:")
print(df.describe())

print("\n Data types:")
print(df.dtypes)

print("\n Missing values:")
print(df.isnull().sum())

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
# Create folder for plots
plot_dir = "/content/tourism_project_CICD/eda_outputs"
os.makedirs(plot_dir, exist_ok=True)

# Univariate Analysis
for col in df.columns:
    if df[col].dtype != 'object':  # numeric columns only
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{col}_distribution.png")
        plt.show()
        plt.close()


# Bivariate Analysis
pairplot = sns.pairplot(df, diag_kind="kde", hue="Engine_Condition")
pairplot.savefig(f"{plot_dir}/pairplot.png")
plt.show()
plt.close()

# Multivariate Analysis (Correlation Heatmap)
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Engine Parameters")
plt.tight_layout()
plt.savefig(f"{plot_dir}/correlation_heatmap.png")
plt.show()
plt.close()

# Insights
print("\n📈 Correlation with Engine_Condition:")
print(df.corr()["Engine_Condition"].sort_values(ascending=False))

print("\n💡 Observations:")
print("- Strong correlations with Engine_Condition may indicate predictive features.")
print("- High coolant/oil temperature or abnormal pressures could signal faults.")
print("- RPM combined with fuel pressure may show combustion efficiency.")

# -----------------------------
# Upload Plots to Hugging Face (optional)
# -----------------------------
for file in os.listdir(plot_dir):
    local_path = os.path.join(plot_dir, file)
    repo_path = f"eda_outputs/{file}"  # folder inside HF repo
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"Uploaded {file} to Hugging Face dataset repo at {repo_path}")
