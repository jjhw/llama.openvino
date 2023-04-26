from torch.onnx import _export as torch_onnx_export
import torch
from transformers import LlamaForCausalLM
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
parser.add_argument('-m', '--model_path',  required=True, type=str,
                    help='Required. orignal model path')
parser.add_argument('-o', '--onnx_path',  required=True, type=str,
                    help='Required. onnx model path')
args = parser.parse_args()

model = LlamaForCausalLM.from_pretrained(args.model_path)
inputs = {
    "input_ids": torch.randint(50, (1, 5)),
    "attention_mask": torch.randint(50, (1, 5)),
}
model.eval()
with torch.no_grad():
    torch_onnx_export(model, inputs, args.onnx_path,input_names=list(inputs), dynamic_axes={
                          list(inputs)[0]: {0: 'batch_size', 1: 'sequence_length'},
                          list(inputs)[1]: {0: 'batch_size', 1: 'sequence_length'}}
                      )