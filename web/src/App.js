import React from 'react';
import JsmpegPlayer from './JsmpegPlayer';
import './App.css';

function App() {
  let jsmpegPlayer = null;

  return (
    <div className="App">
      <header className="App-header">
        <JsmpegPlayer
          wrapperClassName="video-wrapper"
          videoUrl="ws://192.168.92.22:8081" //get using API!!
          onRef={ref => jsmpegPlayer = ref}
        />
        <JsmpegPlayer
          wrapperClassName="video-wrapper"
          videoUrl="ws://192.168.92.22:8084"
          onRef={ref => jsmpegPlayer = ref}
        />
      </header>
    </div>
  );
}

export default App;
