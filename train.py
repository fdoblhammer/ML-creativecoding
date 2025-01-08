from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
# Try to add "device=0" for GPU training, if you have a NVIDIA card or Apple Silicon Chip (M1,M2,M3,M4)
results = model.train(data="CallGesture/data.yaml", epochs=100, imgsz=640)