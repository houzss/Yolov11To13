from ultralytics import YOLO
import torch


def train():
  # model = YOLO('yolov13n.yaml')
  model = YOLO('yolov13n_zhs.pt')
  # model = YOLO('yolov8n.pt')

  # 需要重置全连接层，输出维度靠齐

  # Train the model
  results = model.train(
    # data='coco128.yaml',
    data='wzdkBRT.yaml',
    epochs=300,
    batch=16,
    imgsz=640,
    scale=0.5,  # S:0.9; L:0.9; X:0.9
    mosaic=1.0,
    mixup=0.0,  # S:0.05; L:0.15; X:0.2
    copy_paste=0.1,  # S:0.15; L:0.5; X:0.6
    device="0",
  )

  # Evaluate model performance on the validation set
  # metrics = model.val('coco128.yaml')

  # Perform object detection on an image
  # results = model("path/to/your/image.jpg")
  # results[0].show()
  # torch.save(model, 'test.pt')
  # torch.save(model, model_file)

if __name__ == '__main__':
  train()



