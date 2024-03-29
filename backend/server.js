const express = require('express');
const cors = require('cors');
const startStream = require('./startStream');
const { execSync } = require('child_process');
const db = require('better-sqlite3')('./sqlite/sqlite.db');
const path = require('path');

const createCameraTable = db.prepare('CREATE TABLE IF NOT EXISTS camera( port INT(32) PRIMARY KEY NOT NULL, name VARCHAR(255) NOT NULL, path VARCHAR(255) NOT NULL, httpPort INT(32) NOT NULL, wsPort INT(32) NOT NULL, ffmpegPort INT(32) NOT NULL, lat VARCHAR(255), lon VARCHAR(255), camFps INT(32) NOT NULL, camResolution INT(32) NOT NULL, bv VARCHAR(255) NOT NULL, maxRate VARCHAR(255) NOT NULL, bufSize VARCHAR(255) NOT NULL)');
createCameraTable.run();
const createConfigTable = db.prepare('CREATE TABLE IF NOT EXISTS config( zoom INT(32) NOT NULL, latitude VARCHAR(255) NOT NULL, longitude VARCHAR(255) NOT NULL, bearing INT(32) NOT NULL, pitch INT(32) NOT NULL)');
createConfigTable.run();

const cameras = [];
const getConfigs = db.prepare('SELECT * FROM config LIMIT 1');
let configs = getConfigs.get();

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
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'pug');

app.get('/fetchCameras', (req, res) => {
  console.log("fetchCameras");
  res.send(cameras);
});

app.get('/viewCamera', (req, res) => {
  console.log("viewCamera");
  const data = decodeURI(req.query.url);
  res.render('viewCamera', { url: data });
});

app.get('/fetchConfigs', (req, res) => {
  console.log("fetchConfigs");
  if(configs) res.send(configs);
  else res.send({notset: true});
});

app.post('/setCameraPos', (req, res) => {
  console.log("setCameraPos");
  const updateCamera = db.prepare('UPDATE camera SET lat=@lat, lon=@lon WHERE port=@port');

  updateCamera.run({
    lat: req.body.lat,
    lon: req.body.lon,
    port: req.body.port
  });

  cameras.map((camera, index) => {
    if(camera.port == req.body.port) {
      cameras[index].lon = req.body.lon;
      cameras[index].lat = req.body.lat;
    }
  });

  res.send(cameras.filter(camera => camera.port == req.body.port));
});

app.post('/setConfigs', (req, res) => {
  console.log("setConfigs");
  const newConfigs = {
    zoom: req.body.zoom,
    latitude: req.body.latitude,
    longitude: req.body.longitude,
    bearing: req.body.bearing,
    pitch: req.body.pitch
  }

  const clearTable = db.prepare('DELETE FROM config'); 
  clearTable.run();
  const updateConfigs = db.prepare('INSERT INTO config (zoom, latitude, longitude, bearing, pitch) VALUES (@zoom, @latitude, @longitude, @bearing, @pitch)');
  updateConfigs.run(newConfigs);

  configs = newConfigs;

  res.send(configs);
});

app.post('/updateCamera', (req, res) => {
  console.log("updateCamera");
  const updateCamera = db.prepare('UPDATE camera SET name=@name, path=@path, httpPort=@httpPort, wsPort=@wsPort, ffmpegPort=@ffmpegPort, lat=@lat, lon=@lon, camFps=@camFps, camResolution=@camResolution, bv=@bv, maxRate=@maxRate, bufSize=@bufSize WHERE port=@port');
  updateCamera.run(req.body);

  cameras.map((camera, index) => {
    if(camera.port == req.body.port) {
      cameras[index] = req.body;
    }
  });

  res.send(cameras.filter(camera => camera.port == req.body.port));
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
