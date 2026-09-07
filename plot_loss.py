import json
import glob
import matplotlib.pyplot as plt

# Find the most recent log file
log_files = sorted(glob.glob("logs/*.json"))
if not log_files:
    print("No logs found.")
    exit()

latest_log = log_files[-1]
with open(latest_log, 'r') as f:
    data = json.load(f)

# Extract steps and loss values
steps = [entry["step"] for entry in data["metrics"]]
losses = [entry["loss"] for entry in data["metrics"]]

# Plot the trajectory
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, label="Training Loss", color="blue", linewidth=2)
plt.title(f"Loss Trajectory: {data['timestamp']}")
plt.xlabel("Training Steps")
plt.ylabel("Cross Entropy Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()