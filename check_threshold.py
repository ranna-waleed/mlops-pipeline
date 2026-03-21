# check_threshold.py
import mlflow
import sys
import os

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
THRESHOLD = 0.85

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Read run ID from the artifact file
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking run ID: {run_id}")

# Fetch metrics from MLflow
client = mlflow.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0)

print(f"Accuracy from MLflow: {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f" FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
    sys.exit(1)  # This will FAIL the GitHub Actions job
else:
    print(f" PASSED: Accuracy {accuracy:.4f} meets threshold {THRESHOLD}")
    sys.exit(0)