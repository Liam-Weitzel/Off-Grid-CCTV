from ultralytics import YOLO

# Train
model = YOLO('yolov8n.pt')
model.train(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=200, imgsz=480)

# # Run trained model on images
# model = YOLO('./saved_runs/doumo_200_epochs/weights/best.pt')#.load('')
# results = model(['test1.jpg', 'test2.jpg'])
# for result in results:
#     boxes = result.boxes
#     masks = result.masks
#     keypoints = result.keypoints
#     probs = result.probs
#     result.show()
#     result.save(filename='result-'+result.path)

# # Validate trained model
# model = YOLO('yolov8n.pt')#.load('')
# metrics = model.val(data='data/labeled/duomo/data.yaml', imgsz=640)
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category

# # Run model on live stream
# import cv2
# cap = cv2.VideoCapture("rtp://127.0.0.1:1234")
# model = YOLO('./saved_runs/kielce_university_of_technology_100/weights/best.pt')
# while True:
#     ret, frame = cap.read()
#     detections = model(frame)[0]
#     for data in detections.boxes.data.tolist():
#         confidence = data[4]
#         if float(confidence) < 0.01:
#             continue
#         xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
#         cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), (255, 0, 0), 2)
#     cv2.imshow('video', frame)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27: 
#         break

# # Tune hyperparameters
# model = YOLO('yolov8n.pt')
# model.tune(data='data/toggenburg_alpaca_ranch/data.yaml', epochs=30, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)