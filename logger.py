import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

class ExperimentLogger:
    """ Handles asynchronous JSON logging and plotting for ML experiments. """
    def __init__(self, config, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f"run_{self.timestamp}.json")
        
        self.run_data = {
            "timestamp": self.timestamp,
            "hyperparameters": config,
            "metrics": []
        }
        
        self._save()
        print(f"[Logger] Tracking experiment in: {self.log_file}")

    def log_model_metadata(self, param_count, device):
        self.run_data["total_parameters"] = param_count
        self.run_data["device"] = str(device)
        self._save()

    def log_step(self, step, loss):
        self.run_data["metrics"].append({
            "step": step,
            "loss": loss
        })
        self._save()
        
    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.run_data, f, indent=4)

    def generate_plot(self):
        """ Generates and saves a PNG of the training loss curve. """
        if not self.run_data["metrics"]:
            return

        steps = [entry["step"] for entry in self.run_data["metrics"]]
        losses = [entry["loss"] for entry in self.run_data["metrics"]]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, label="Training Loss", color="#1f77b4", linewidth=2, marker="o", markersize=4)
        
        # Add visual metadata
        plt.title(f"Loss Trajectory | Final Loss: {losses[-1]:.4f}")
        plt.suptitle(f"Run: {self.timestamp}", fontsize=10, color="gray")
        plt.xlabel("Training Steps")
        plt.ylabel("Cross Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # Save to disk
        plot_file = os.path.join(self.log_dir, f"plot_{self.timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close() # Free system memory
        print(f"[Logger] Loss plot saved to: {plot_file}")