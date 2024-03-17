import { React, useState, useEffect } from 'react';
import JsmpegPlayer from './jsmpegPlayer.js';
import Draggable from 'react-draggable';
import { Resizable } from 'react-resizable';

function PopOutVideo(props) {
  const [windowSize, setWindowSize] = useState({width: 320, height: 210});

  const onResize = (event, {node, size, handle}) => {
    setWindowSize({width: size.width, height: size.height});
  };

  return (
  <Draggable cancel=".react-resizable-handle">
    <Resizable
    height={windowSize.height}
    width={windowSize.width}
    onResize={onResize}
    minConstraints={[106, 70]}
    maxConstraints={[1280, 840]}
    lockAspectRatio={true}
    >
      <div
      className="video-pop-out"
      style={{width: windowSize.width + 'px', height: windowSize.height + 'px'}}
      >
        <button className="close-popout-button" type="button" aria-label="Close popup" aria-disabled="false" onClick={()=>props.closePopOutVideo(props.httpPort)}>X</button>
        <JsmpegPlayer
          wrapperClassName="video-wrapper"
          videoUrl={'ws://' + props.backendIP + ':' + props.httpPort}
        />
      </div>
    </Resizable>
  </Draggable>
  );
}

export default PopOutVideo;
