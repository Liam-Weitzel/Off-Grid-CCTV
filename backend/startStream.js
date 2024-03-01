const { spawn } = require('child_process');
const filter = require('stream-filter');

const startStream = (camPort, camFps, frameSize, httpPort, wsPort, streamPort, streamSecret, ffmpegArgs) => {
  
  const ffmpegStream = spawn('ffmpeg', ffmpegArgs);
  ffmpegStream.stdin.setEncoding('binary');

  const detectionStream = spawn('node', ['detection.js', camPort, camFps, frameSize, httpPort, wsPort, streamPort, streamSecret]);
  detectionStream.stdout.setDefaultEncoding('binary');
  
  require('./createStreamServer')(httpPort, streamPort, streamSecret);

  detectionStream.stdout.on('data', (data) => {
    //console.log('opencv stdout: ' + data.toString());
  });

  detectionStream.stderr.on('data', (data) => {
    console.log(camPort + ': ' + data.toString());
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

  return { ffmpegStream, detectionStream };
}

module.exports = startStream;
