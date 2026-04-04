"""
Export PyTorch model weights to JSON for browser inference.

Output format:
{
  "layers": [
    { "weight": [[...], ...], "bias": [...] },
    { "weight": [[...], ...], "bias": [...] },
    { "weight": [[...], ...], "bias": [...] }
  ]
}
"""
import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(__file__))

from network import HanabiNet


def export_weights(ckpt_path, output_path):
    model = HanabiNet()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.eval()

    layers = []
    # net.0 = Linear(124, 128), net.2 = Linear(128, 128), net.4 = Linear(128, 68)
    for i in [0, 2, 4]:
        layer = model.net[i]
        layers.append({
            'weight': layer.weight.detach().numpy().tolist(),
            'bias': layer.bias.detach().numpy().tolist(),
        })

    data = {'layers': layers}
    with open(output_path, 'w') as f:
        json.dump(data, f)

    # Report size
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported {len(layers)} layers to {output_path} ({size_kb:.0f} KB)")


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'policy_net.pt')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'model_weights.json')

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Run train.py first.")
        sys.exit(1)

    export_weights(ckpt_path, output_path)


if __name__ == '__main__':
    main()
