import torch
import argparse
from fms.models import get_model

parser = argparse.ArgumentParser(description="Convert Hugging Face model to FMS format")
parser.add_argument('--model_variant', type=str, required=True, help='Model variant (e.g., llama3_8b)')
parser.add_argument('--hf_path', type=str, required=True, help='Path to the Hugging Face model')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the converted FMS model')
args = parser.parse_args()

variant = args.model_variant[5:].replace('_', '-')
fms_model = get_model(
    architecture="llama",
    variant=variant,
    model_path=args.hf_path,
    source='hf'
)
print(f"Loaded FMS model from {args.hf_path}")

state_dict = {"model_state": fms_model.state_dict()}
torch.save(state_dict, args.save_path)
print(f"Saved FMS model to {args.save_path}")

