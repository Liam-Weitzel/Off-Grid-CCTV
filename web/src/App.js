import { React, useEffect, useState } from 'react';
import JsmpegPlayer from './components/JsmpegPlayer';
import './App.css';

const backendIP = '192.168.92.22'
const apiPort = '8080'

function App() {
  let jsmpegPlayer = null;
  const [cameras, setCameras] = useState([]);

  const searchCameras = async (q) => {
    const response = await fetch (`http://${backendIP}:${apiPort}/searchCameras?` + new URLSearchParams({q}));
    const data = await response.json();
    setCameras(data);
  };

  return (
    <div className="App">
      <header className="App-header">
      <input 
        type="text"
        placeholder="search"
        onChange={(e) => searchCameras(e.target.value)}
      /> 

      {cameras.map((camera) => (
        <div key={camera.secret}>
        <p> {camera.secret} </p>
        <JsmpegPlayer
          wrapperClassName="video-wrapper"
          videoUrl={'ws://' + backendIP + ':' + camera.httpPort}
          onRef={ref => jsmpegPlayer = ref}
        />
        </div>
      ))}
      {cameras.length === 0 && <p>No cameras found! </p>}
      </header>
    </div>
  );
}

export default App;
