const WebSocket = require('ws');
const http = require('http');

const createStreamServer = (httpPort, streamPort, streamSecret) => {
  // HTTP server
  const httpServer = http.createServer();
  httpServer.listen(httpPort, () => {
    console.log('HTTP server listening on port ' + httpPort);
  });

  // Websocket server
  const socketServer = new WebSocket.Server({ server: httpServer });

  socketServer.connectionCount = 0;

  socketServer.on('connection', (socket, upgradeReq) => {
    socketServer.connectionCount++;

    console.log(`New WebSocket Connection: 
      ${(upgradeReq || socket.upgradeReq).socket.remoteAddress}
      ${(upgradeReq || socket.upgradeReq).headers['user-agent']}
      (${socketServer.connectionCount} total)`
    );

    socket.on('close', () => {
      socketServer.connectionCount--;
      console.log('Disconnected WebSocket (' + socketServer.connectionCount + ' total)');
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
      console.log(`Failed Stream Connection: ${request.socket.remoteAddress}:${request.socket.remotePort}`);
      response.end();
    }

    response.connection.setTimeout(0);

    console.log(`Stream Connected: ${request.socket.remoteAddress}:${request.socket.remotePort}`);

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
}

module.exports = createStreamServer;
