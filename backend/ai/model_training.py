from ultralytics import YOLO

# # Train
# model = YOLO('yolov8n.pt')
# model.train(data='data/duomo/data.yaml', epochs=200, imgsz=640)

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

# Run model on live stream
import cv2
cap = cv2.VideoCapture("http://88.53.197.250/mjpg/video.mjpg")
while True:
    ret, frame = cap.read()
    model = YOLO('./saved_runs/doumo_200_epochs/weights/best.pt')#.load('')
    detections = model(frame)[0]
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < 0.2:
            continue
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), (255, 0, 0), 2)
    cv2.imshow('video', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
