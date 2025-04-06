from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model on your dataset
results = model.train(
    data='./datasets/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device='cpu'
)

# Optional: Print the results
print("Training Complete!")