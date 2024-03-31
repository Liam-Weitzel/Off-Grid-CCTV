const path = require('path');
const fs = require('fs');

const { cv, grabFrames } = require('./opencv-helpers');

camPort = parseInt(process.argv[2]);
camFps = parseInt(process.argv[3]);
frameSize = parseInt(process.argv[4]);
httpPort = parseInt(process.argv[5]);
wsPort = parseInt(process.argv[6]);
streamPort = parseInt(process.argv[7]);
streamSecret = process.argv[8];

classNames = {
  0: 'person',
  1: 'bicycle',
  2: 'car',
  3: 'motorcycle',
  4: 'truck',
  5: 'cat',
  6: 'dog',
  7: 'horse',
  8: 'sheep',
  9: 'cow',
  10: 'bear',
  11: 'zebra',
};

if (!cv.xmodules.dnn) {
  throw new Error('exiting: opencv4nodejs compiled without dnn module');
}

// set stdout encoding to 'binary'
process.stdout.setDefaultEncoding('binary');

const modelPath = path.resolve(__dirname, 'ai/obj_detection_model/frozen_inference_graph.pb');
const configPath = path.resolve(
  __dirname,
  'ai/obj_detection_model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
);

if (!fs.existsSync(modelPath) || !fs.existsSync(configPath)) {
  console.log('could not find tensorflow object detection model');
  console.log('download the model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#use-existing-config-file-for-your-model');
  throw new Error('exiting: could not find tensorflow object detection model');
}

// initialize tensorflow darknet model from modelFile
const net = cv.readNetFromTensorflow(modelPath, configPath);

// set webcam interval
const camInterval = 1000 / camFps;

const objectDetect = (img) => {
  // object detection model works with 300 x 300 images
  const size = new cv.Size(300, 300);
  const vec3 = new cv.Vec(0, 0, 0);

  // network accepts blobs as input
  const inputBlob = cv.blobFromImage(img, 1, size, vec3, true, true);
  net.setInput(inputBlob);

  // forward pass input through entire network, will return
  // classification result as 1x1xNxM Mat
  const outputBlob = net.forward();

  // get height and width from the image
  const [imgHeight, imgWidth] = img.sizes;
  const numRows = outputBlob.sizes.slice(2, 3);

  for (let y = 0; y < numRows; y += 1) {
    const confidence = outputBlob.at([0, 0, y, 2]);
    if (confidence > 0.8) {
      const classId = outputBlob.at([0, 0, y, 1]);
      const className = classNames[classId];
      /* const boxX = imgWidth * outputBlob.at([0, 0, y, 3]);
      const boxY = imgHeight * outputBlob.at([0, 0, y, 4]);
      const boxWidht = imgWidth * outputBlob.at([0, 0, y, 5]);
      const boxHeight = imgHeight * outputBlob.at([0, 0, y, 6]);

      const pt1 = new cv.Point(boxX, boxY);
      const pt2 = new cv.Point(boxWidht, boxHeight);
      const rectColor = new cv.Vec(23, 230, 210);
      const rectThickness = 2;
      const rectLineType = cv.LINE_8;

      // draw the rect for the object
      img.drawRectangle(pt1, pt2, rectColor, rectThickness, rectLineType);

      const text = `${className} ${confidence.toFixed(5)}`;
      const org = new cv.Point(boxX, boxY + 15);
      const fontFace = cv.FONT_HERSHEY_SIMPLEX;
      const fontScale = 0.5;
      const textColor = new cv.Vec(123, 123, 255);
      const thickness = 2;

      // put text on the object
      img.putText(text, org, fontFace, fontScale, textColor, thickness); */
      console.warn(className);
    }
  }

  // write the jpg binary data to stdout
  process.stdout.write(cv.imencode('.jpg', img).toString('binary'));
};

const runWebcamObjectDetect = (src, objectDetect) =>
  grabFrames(src, camInterval, (frame) => {
    const frameResized = frame.resizeToMax(frameSize);

    // detect objects
    objectDetect(frameResized);
  });

runWebcamObjectDetect(camPort, objectDetect);
