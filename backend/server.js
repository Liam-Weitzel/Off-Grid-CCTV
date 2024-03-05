const express = require('express');
const cors = require('cors');
const startStream = require('./startStream');
const { execSync } = require('child_process');
const db = require('better-sqlite3')('../sqlite/sqlite.db'); //different path on prod

const configs = { //TODO: move to db
  latitude: 36.90453150945084,
  longitude: 15.013785520105046,
  zoom: 16,
  bearing: -50,
  pitch: 0,
  id: 'sky',
  type: 'sky',
  paint: {
    'sky-type': 'atmosphere',
    'sky-atmosphere-sun': [0.0, 0.0],
    'sky-atmosphere-sun-intensity': 15
  }};

const createTable = db.prepare('CREATE TABLE IF NOT EXISTS camera( port INT(32) PRIMARY KEY NOT NULL, name VARCHAR(255) NOT NULL, path VARCHAR(255) NOT NULL, httpPort INT(32) NOT NULL, wsPort INT(32) NOT NULL, ffmpegPort INT(32) NOT NULL, lat VARCHAR(255), lon VARCHAR(255), camFps INT(32) NOT NULL, camResolution INT(32) NOT NULL, bv VARCHAR(255) NOT NULL, maxRate VARCHAR(255) NOT NULL, bufSize VARCHAR(255) NOT NULL)');
createTable.run();

const cameras = [];

const stdout = execSync('v4l2-ctl --list-devices').toString();
const usbCameraPaths = [];
const lines = stdout.split("\n");
lines.forEach(line => {
  if (line.includes("/dev/video")) { 
    usbCameraPaths.push(line.trim());
  }
});

usbCameraPaths.forEach(usbCameraPath => {
  const isCameraInTable = db.prepare('SELECT * FROM camera WHERE port = ?').bind(parseInt(usbCameraPath.replace(/[^0-9]/g, '')));
  const res = isCameraInTable.get();
  if(!res) {
    const lastPort = db.prepare('SELECT COALESCE((SELECT MAX(ffmpegPort) FROM camera), 8080) AS lastPort').get().lastPort;
    console.log(lastPort);
    const camera = {
      port: parseInt(usbCameraPath.replace(/[^0-9]/g, '')),
      name: ('camera'+parseInt(usbCameraPath.replace(/[^0-9]/g, ''))).toString(),
      path: usbCameraPath,
      httpPort: parseInt(lastPort+1),
      wsPort: parseInt(lastPort+2),
      ffmpegPort: parseInt(lastPort+3),
      lat: null,
      lon: null,
      camFps: 30,
      camResolution: 640,
      bv: '1000k',
      maxRate: '1000k',
      bufSize: '500k',
    };

    cameras.push(camera);
    const insertCamera = db.prepare('INSERT INTO camera (port, name, path, httpPort, wsPort, ffmpegPort, lat, lon, camFps, camResolution, bv, maxRate, bufSize) VALUES (@port, @name, @path, @httpPort, @wsPort, @ffmpegPort, @lat, @lon, @camFps, @camResolution, @bv, @maxRate, @bufSize)');
    insertCamera.run(camera);
  } else {
    cameras.push(res);
  }
});

const app = express();
app.use(cors());
app.use(express.json());

app.get('/fetchCameras', (req, res) => {
  res.send(cameras);
});

app.get('/fetchConfigs', (req, res) => {
  res.send(configs);
});

app.listen(8080, () => console.log('api localhost:8080'))

cameras.forEach((camera) => {
  const { ffmpegStream, detectionStream } = startStream(camera.port, camera.camFps, camera.camResolution, camera.httpPort, camera.wsPort, camera.ffmpegPort, camera.name.toString(), ['-f', 'image2pipe', '-', '-i', '-', '-f', 'mpegts', '-c:v', 'mpeg1video', '-b:v', camera.bv.toString(), '-maxrate:v', camera.maxRate.toString(), '-bufsize', camera.bufSize.toString(), '-an', `http://localhost:${camera.ffmpegPort}/${camera.name}`]); 
 
  detectionStream.on('exit', () => {
    //exit gracefully
  });
  ffmpegStream.on('exit', () => {
    //exit gracefully
  });
});
