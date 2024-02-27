const { spawn, exec } = require('child_process');

// Get all connected cameras from USB devices
exec('v4l2-ctl --list-devices', (err, stdout, stderr) => {
  if (err) {
    console.error(`exec error: ${err}`);
    return;
  }
  
  const cameras = [];
  const lines = stdout.split("\n");
  lines.forEach(line => {
    if (line.includes("/dev/video")) { 
      cameras.push(line.trim());
    }
  });

  cameras.forEach(camera => {
    console.log(camera)
    //check if its a working camera -> how?
    //what kind of camera is it? what ffmpeg settings do we use?

    spawn('node', ['streamServer.js', 1, 30, 640, 8080, 8081, 8082, 'rst']);
    // spawn('node', ['streamServer.js', camPort, camFps, frameSize, httpPort, wsPort, streamPort, streamSecret]);
  });
});
