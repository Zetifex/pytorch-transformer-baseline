import json
import os
from datetime import datetime

class ExperimentLogger:
    """ Handles asynchronous JSON logging for ML experiments. """
    def __init__(self, config, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate a unique filename based on the current time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f"run_{timestamp}.json")
        
        # Initialize the base tracking dictionary
        self.run_data = {
            "timestamp": timestamp,
            "hyperparameters": config,
            "metrics": []
        }
        
        # Create the initial file
        self._save()
        print(f"[Logger] Tracking experiment in: {self.log_file}")

    def log_model_metadata(self, param_count, device):
        """ Logs hardware and architectural scale before training starts. """
        self.run_data["total_parameters"] = param_count
        self.run_data["device"] = str(device)
        self._save()

    def log_step(self, step, loss):
        """ Appends step metrics incrementally during the training loop. """
        self.run_data["metrics"].append({
            "step": step,
            "loss": loss
        })
        self._save()
        
    def _save(self):
        """ Internal method to safely overwrite the JSON file. """
        with open(self.log_file, 'w') as f:
            json.dump(self.run_data, f, indent=4)