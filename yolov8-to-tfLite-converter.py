from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  

# Export directly to TFLite format
model.export(format="tflite")

print("✅ YOLOv8 successfully exported to TFLite!")


