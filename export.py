import torch
import torch.onnx

input_model_path = input('Enter the path of the model to export: ')
output_model_path = input('Save ONNX model as: ')

model = torch.load(input_model_path)
model.eval()

dummy_input = torch.randn(1, 3, 64, 64)

torch.onnx.export(model, dummy_input, output_model_path, export_params=True, opset_version=12)
