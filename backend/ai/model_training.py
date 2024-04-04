from ultralytics import YOLO

# To train
model = YOLO('yolov8n.pt')
model.train(data='data/labeled/duomo/data.yaml', epochs=1, imgsz=640)

# To run trained model
model = YOLO('yolov8n.pt').load('./runs/detect/train3/weights/best.pt')
results = model(['test1.jpg', 'test2.jpg'])
for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    result.show()
    result.save(filename='result-'+result.path)

# Validate trained model
metrics = model.val(data='data/labeled/duomo/data.yaml', imgsz=640)
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
