from ultralytics import YOLO
import torch

# model = YOLO('yolov13n.yaml')
model = YOLO('yolov13n.pt')

# Train the model
results = model.train(
  data='coco.yaml',
  epochs=600,
  batch=256,
  imgsz=640,
  scale=0.5,  # S:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; L:0.5; X:0.6
  device="0,1,2,3",
)

# Evaluate model performance on the validation set
metrics = model.val('coco.yaml')

# Perform object detection on an image
results = model("path/to/your/image.jpg")
results[0].show()
torch.save(model, '')