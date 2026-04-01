"""
Export trained weights to JSON for TypeScript consumption.
"""
import sys
import json
import numpy as np
from train import load_weights

def export(weights_path, output_path):
    weights = load_weights(weights_path)
    # Convert to lists for JSON
    data = {k: v.tolist() for k, v in weights.items()}
    with open(output_path, 'w') as f:
        json.dump(data, f)

    # Print stats
    total_params = sum(v.size for v in weights.values())
    file_size = len(json.dumps(data))
    print(f"Exported {total_params} parameters to {output_path}")
    print(f"File size: {file_size / 1024:.0f} KB")


if __name__ == '__main__':
    weights_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/weights_final.json'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '../src/engine/model_weights.json'
    export(weights_path, output_path)
