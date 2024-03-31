from ultralytics import YOLO

# To train
model = YOLO('yolov8n.pt')
model.train(data='data/data.yaml', epochs=1, imgsz=640)

# # To run trained model
# model = YOLO('yolov8n.pt').load('./best.pt')
# results = model(['test1.jpg', 'test2.jpg'])
# for result in results:
#     boxes = result.boxes
#     masks = result.masks
#     keypoints = result.keypoints
#     probs = result.probs
#     result.show()
#     result.save(filename='result-'+result.path)
