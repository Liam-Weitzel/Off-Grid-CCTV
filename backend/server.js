const express = require('express');
const cors = require('cors');
const startStream = require('./startStream');
const { exec } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());

/* app.set('views', __dirname);
app.set('view engine', 'pug');

app.get('/8081', async (req, res) => {
  res.render('index', { title: 'Object Detection on Live Stream', url: 'ws://localhost:8081'});
}); */

app.get('/getWebSockets', async (req, res) => {
  res.send('ws://localhost:8081');
});

app.listen(8080, () => console.log('api localhost:8080'))

const { ffmpegStream, detectionStream } = startStream(1, 30, 640, 8081, 8082, 8083, 'camera1', ['-f', 'image2pipe', '-', '-i', '-', '-f', 'mpegts', '-c:v', 'mpeg1video', '-b:v', '1000k', '-maxrate:v', '1000k', '-bufsize', '500k', '-an', `http://localhost:8083/camera1`]);

//Start a stream for all plugged in cameras
exec('v4l2-ctl --list-devices', (err, stdout) => {
  if (err) {
    console.error(`exec error: ${err}`);
    return;
  }

  const usbCameraPaths = [];
  const lines = stdout.split("\n");
  lines.forEach(line => {
    if (line.includes("/dev/video")) { 
      usbCameraPaths.push(line.trim());
    }
  });

  usbCameraPaths.forEach(usbCameraPath => {
    console.log(usbCameraPath);
  });
});
