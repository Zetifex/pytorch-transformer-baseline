import json
import os
import glob

# Find all JSON files in the logs directory
log_files = glob.glob("logs/*.json")

if not log_files:
    print("No logs found.")
else:
    print(f"Found {len(log_files)} experiment(s):\n")
    
    for file in log_files:
        with open(file, 'r') as f:
            data = json.load(f)
            
            # Extract basic data
            timestamp = data.get("timestamp", "Unknown")
            params = data.get("total_parameters", 0)
            metrics = data.get("metrics", [])
            
            # Find the final loss
            if metrics:
                final_loss = metrics[-1]["loss"]
                print(f"Run: {timestamp} | Params: {params/1e6:.2f}M | Final Loss: {final_loss:.4f}")
            else:
                print(f"Run: {timestamp} | Incomplete run (No metrics)")