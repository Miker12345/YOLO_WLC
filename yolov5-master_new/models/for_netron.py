import torch
from models.yolo import Model

cfg = "yolov5l.yaml"
# Create model
model = Model(cfg).to("cpu")
x = torch.randn(1, 3, 640, 640).to("cpu")
script_model = torch.jit.trace(model, x)
script_model.save("m.pt")
