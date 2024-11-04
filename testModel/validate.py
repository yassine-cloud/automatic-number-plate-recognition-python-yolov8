from ultralytics import YOLO

# Load a model
model = YOLO("license_plate_detector_GPU_NOv6.pt")

# Customize validation settings
validation_results = model.val(data="../train/v11/v8/data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="cuda:0")