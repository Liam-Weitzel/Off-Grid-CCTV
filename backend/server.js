const { spawn, exec } = require('child_process');
const filter = require('stream-filter');

// Get all connected cameras from USB devices and start streams
exec('v4l2-ctl --list-devices', (err, stdout, stderr) => {
  if (err) {
    console.error(`exec error: ${err}`);
    return;
  }
  
  const usbCameraPaths = []; // use these to generate the configs for each stream
  const lines = stdout.split("\n");
  lines.forEach(line => {
    if (line.includes("/dev/video")) { 
      usbCameraPaths.push(line.trim());
    }
  });

  //TODO: detect what kind of camera it is and what we settings we should use
  
  //Start all processes for each camera and store a pointer
  const ffmpegStreams = [];
  const detectionStreams = [];
  const cameraConfigs = [];
  usbCameraPaths.forEach(usbCameraPath => {
    if(usbCameraPath=="/dev/video1") { //temp hardcoded configs and camera
      const camPort = 1;
      const camFps = 30;
      const frameSize = 640;
      const httpPort = 8080;
      const wsPort = 8081;
      const streamPort = 8082;
      const streamSecret = 'camera1';
      const ffmpegArgs = [
              '-f',
              'image2pipe',
              '-',
              '-i',
              '-',
              '-f',
              'mpegts',
              '-c:v',
              'mpeg1video',
              // "-q",
              // "10",
              '-b:v',
              '1000k',
              '-maxrate:v',
              '1000k',
              '-bufsize',
              '500k',
              '-an',
              `http://localhost:${streamPort}/${streamSecret}`
      ];

      cameraConfigs.push([camPort, camFps, frameSize, httpPort, wsPort, streamPort, streamSecret, ffmpegArgs]);

      const ffmpegStream = spawn('ffmpeg', ffmpegArgs);
      ffmpegStream.stdin.setEncoding('binary');
      ffmpegStreams.push(ffmpegStream);

      const detectionStream = spawn('node', ['detection.js', camPort, camFps, frameSize, httpPort, wsPort, streamPort, streamSecret]);
      detectionStream.stdout.setDefaultEncoding('binary');
      detectionStreams.push(detectionStream); 

      require('./streamServer')(httpPort, streamPort, streamSecret);
    }
  });

  for (let index = 0; index < detectionStreams.length; index++) {
    const detectionStream = detectionStreams[index];
    const ffmpegStream = ffmpegStreams[index];
    const cameraConfig = cameraConfigs[index];

    detectionStream.stdout.on('data', (data) => {
      //console.log('opencv stdout: ' + data.toString());
    });

    detectionStream.stderr.on('data', (data) => {
      console.log('detected on ' + cameraConfig[0] + ": " + data.toString());
    });

    detectionStream.on('exit', (code) => {
      console.log('opencv exited ' + code.toString());
    });

    detectionStream.stdout
      .pipe(
        filter((data) => {
                // filter opencv output, example: '[ INFO:0] Initialize OpenCL runtime...'
                const regex = /\[\sINFO:\d\]\s/gm;
                const isImgData = !regex.test(data);
                return isImgData;
        })
      )
      .pipe(ffmpegStream.stdin);

    ffmpegStream.stdout.on('data', (data) => {
      //console.log('stdout: ' + data.toString());
    });

    ffmpegStream.stderr.on('data', (data) => {
      //console.log('ffmpeg stderr: ' + data.toString());
    });

    ffmpegStream.on('exit', (code) => {
      console.log('ffmpeg exited ' + code.toString());
    });
  }

});

