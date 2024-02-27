const express = require('express');
const WebSocket = require('ws');
const http = require('http');
const path = require('path');
const ngrok = require('ngrok');
const { spawn } = require('child_process');

camPort = parseInt(process.argv[2]);
camFps = parseInt(process.argv[3]);
frameSize = parseInt(process.argv[4]);
httpPort = parseInt(process.argv[5]);
wsPort = parseInt(process.argv[6]);
streamPort = parseInt(process.argv[7]);
streamSecret = process.argv[8];

// App parameters
const app = express();
app.set('port', httpPort);
app.use(express.static(path.join(__dirname, 'public')));

// View engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'pug');

// Default Websocket URL
let wsUrl = `ws://localhost:${httpPort}`;

// HTTP server
const httpServer = http.createServer(app);
httpServer.listen(app.get('port'), '0.0.0.0', () => {
	console.log('HTTP server listening on port ' + app.get('port'));
});

app.get('/', (req, res) => {
	res.render('index', { title: 'Object Detection on Live Stream', url: wsUrl });
});

// Websocket server
const socketServer = new WebSocket.Server({ server: httpServer });

socketServer.connectionCount = 0;

socketServer.on('connection', (socket, upgradeReq) => {
	socketServer.connectionCount++;

	console.log(
		`New WebSocket Connection: 
    ${(upgradeReq || socket.upgradeReq).socket.remoteAddress}
    ${(upgradeReq || socket.upgradeReq).headers['user-agent']}
    (${socketServer.connectionCount} total)`
	);

	socket.on('close', () => {
		socketServer.connectionCount--;
		console.log(
			'Disconnected WebSocket (' + socketServer.connectionCount + ' total)'
		);
	});
});

socketServer.broadcast = (data) => {
	socketServer.clients.forEach((client) => {
		if (client.readyState === WebSocket.OPEN) {
			client.send(data);
		}
	});
};

// HTTP Server to accept incomming local MPEG-TS Stream from ffmpeg
http.createServer((request, response) => {
	const params = request.url.substr(1).split('/');

	if (params[0] !== streamSecret) {
		console.log(
			`Failed Stream Connection: 
        ${request.socket.remoteAddress}:${request.socket.remotePort}`
		);
		response.end();
	}

	response.connection.setTimeout(0);

	console.log(
		`Stream Connected: 
      ${request.socket.remoteAddress}:${request.socket.remotePort}`
	);

	request.on('data', (data) => {
		socketServer.broadcast(data);
		if (request.socket.recording) {
			request.socket.recording.write(data);
		}
	});

	request.on('end', () => {
		console.log('close');
		if (request.socket.recording) {
			request.socket.recording.close();
		}
	});
})
	.listen(streamPort);

// Start generate streaming
require('./utils/stream')(camPort, camFps, frameSize, httpPort, wsPort, streamPort, streamSecret);

// Get ngrok url for local server
(async function() {
	// IIFE: Immediately Invoked Function Expression
	const httpUrl = await ngrok.connect(httpPort);
	wsUrl = httpUrl.toString().replace(/^https?:\/\//, 'wss://');

	console.log('Ngrok HTTP URL:', httpUrl);
	console.log('Ngrok Websocket URL:', wsUrl);
	console.log();
})().catch((error) => console.log(error.message));

module.exports.app = app;
