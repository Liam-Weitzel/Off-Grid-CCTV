const express = require('express');
const cors = require('cors');
const startStream = require('./startStream');
const { execSync } = require('child_process');
const db = require('better-sqlite3')('../sqlite/sqlite.db'); //different path on prod

const cameras = [];

const stdout = execSync('v4l2-ctl --list-devices').toString();
const usbCameraPaths = [];
const lines = stdout.split("\n");
lines.forEach(line => {
  if (line.includes("/dev/video")) { 
    usbCameraPaths.push(line.trim());
  }
});

currPort = 8080;
usbCameraPaths.forEach(usbCameraPath => {
  const camera = {
    camPort: parseInt(usbCameraPath.replace(/[^0-9]/g, '')),
    camFps: 30,
    camResolution: 640, 
    httpPort: parseInt(currPort+1),
    wsPort: parseInt(currPort+2),
    ffmpegPort: parseInt(currPort+3),
    secret: ('camera'+parseInt(usbCameraPath.replace(/[^0-9]/g, ''))).toString(),
    bv: '1000k',
    maxrate: '1000k',
    bufsize: '500k'
  };
  cameras.push(camera);
  currPort += 3;
});

const app = express();
app.use(cors());
app.use(express.json());

app.get('/searchCameras', (req, res) => {
  const q = req.query.q.toLowerCase() || '';
  const results = cameras.filter(camera => camera.secret.toLowerCase().includes(q));
  res.send(results);
});

app.listen(8080, () => console.log('api localhost:8080'))

cameras.forEach((camera) => {
  const { ffmpegStream, detectionStream } = startStream(camera.camPort, camera.camFps, camera.camResolution, camera.httpPort, camera.wsPort, camera.ffmpegPort, camera.secret, ['-f', 'image2pipe', '-', '-i', '-', '-f', 'mpegts', '-c:v', 'mpeg1video', '-b:v', camera.bv, '-maxrate:v', camera.maxrate, '-bufsize', camera.bufsize, '-an', `http://localhost:${camera.ffmpegPort}/${camera.secret}`]); 
 
  detectionStream.on('exit', () => {
    //exit gracefully
  });
  ffmpegStream.on('exit', () => {
    //exit gracefully
  });
});
