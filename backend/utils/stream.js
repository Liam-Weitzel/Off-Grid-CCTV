const { spawn } = require('child_process');
const filter = require('stream-filter');

const streamArgs = [
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

/* const streamArgs = [
	'-f',
	'image2pipe',
	'-',
	'-i',
	'/dev/video1',
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
	`http://localhost:${server.streamPort}/${server.streamSecret}`
]; */

//ffmpeg -f image2pipe -i - -f mpegts -c:v mpeg1video -b:v 1000k -maxrate:v 1000k -bufsize 500k -an http://localhost:8082/N23y08VnzfDH4Wmf2tXoDyxbwf2rGQJC

const stream = (camPort, camFps, frameSize, httpPort, wsPort, streamPort, streamSecret) => {
        const ffmpegStream = spawn('ffmpeg', streamArgs);
        ffmpegStream.stdin.setEncoding('binary');

        const detectionStream = spawn('node', ['utils/detection.js', camPort, camFps, frameSize, httpPort, wsPort, streamPort, streamSecret]);
        detectionStream.stdout.setDefaultEncoding('binary');

        detectionStream.stdout.on('data', (data) => {
                console.log('opencv stdout: ' + data.toString());
        });

        detectionStream.stderr.on('data', (data) => {
                console.log('opencv stderr: ' + data.toString());
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
                console.log('stdout: ' + data.toString());
        });

        ffmpegStream.stderr.on('data', (data) => {
                console.log('ffmpeg stderr: ' + data.toString());
        });

        ffmpegStream.on('exit', (code) => {
                console.log('ffmpeg exited ' + code.toString());
        });
};

module.exports = stream;
