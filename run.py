import os
import mlflow
import platform

# Platform-specific configuration
if platform.system() == "Windows":
    # Windows-specific settings
    conda_home = r"D:\anaconda3"
    conda_exec = r"D:\anaconda3\Scripts\conda.bat"
else:
    # Linux/macOS settings (for SageMaker)
    conda_home = "/opt/conda"  # Typical SageMaker conda location
    conda_exec = "/opt/conda/bin/conda"

# Configure MLflow
os.environ["MLFLOW_CONDA_HOME"] = conda_home
mlflow.projects.utils._CONDA_EXECUTABLE = conda_exec

experiment_name = "ElasticNet"
entry_point = "Training"

print("\n",os.environ["MLFLOW_CONDA_HOME"])

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Run the MLflow project
mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    experiment_name=experiment_name,
    env_manager="conda"  # Use conda for proper environment management
)