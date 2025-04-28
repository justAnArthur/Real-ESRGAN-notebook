import torch
import torchvision.models as models
from basicsr.archs.rrdbnet_arch import RRDBNet
from onnxruntime import tools

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
input_names, inputs_as_tuple = tools.pytorch_export_helpers.infer_input_info(model, dummy_input)

output_onnx = "RRDBNet.onnx"

torch.onnx.export(model, inputs_as_tuple, "model.onnx", input_names=input_names)