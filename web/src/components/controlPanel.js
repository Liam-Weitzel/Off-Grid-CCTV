import { React, useState, memo} from 'react';
import Draggable from 'react-draggable';

function ControlPanel(props) {
  const [addCameraPopup, setAddCameraPopup] = useState(false);
  const [removeCameraPopup, setRemoveCameraPopup] = useState(false);
  const [refreshCameraPopup, setRefreshCameraPopup] = useState(false);

  const addCamera = () => {
    props.setControlPanelOpen(true)
    setAddCameraPopup(true)
    setRemoveCameraPopup(false)
    setRefreshCameraPopup(false)
  }

  const removeCamera = () => {
    props.setControlPanelOpen(true)
    setAddCameraPopup(false)
    setRemoveCameraPopup(true)
    setRefreshCameraPopup(false)
  }

  const refreshCamera = () => {
    props.setControlPanelOpen(true)
    setAddCameraPopup(false)
    setRemoveCameraPopup(false)
    setRefreshCameraPopup(true)
  }

  const closePopup = () => {
    props.setControlPanelOpen(false)
    setAddCameraPopup(false)
    setRemoveCameraPopup(false)
    setRefreshCameraPopup(false)
  }

  return (
    <div>
    <div className="control-panel mapboxgl-ctrl mapboxgl-ctrl-group-horizontal">
      <button className="add-camera-icon" type="button" aria-label="Add camera" aria-disabled="false" onClick={addCamera}><span className="mapboxgl-ctrl-icon" aria-hidden="true" title="Add camera"></span></button>
      <button className="remove-camera-icon" type="button" aria-label="Remove camera" aria-disabled="false" onClick={removeCamera}><span className="mapboxgl-ctrl-icon" aria-hidden="true" title="Remove camera"></span></button>
      <button className="refresh-camera-icon" type="button" aria-label="Refresh camera" aria-disabled="false" onClick={refreshCamera}><span className="mapboxgl-ctrl-icon" aria-hidden="true" title="Refresh camera"></span></button>
    </div>
    {(addCameraPopup || removeCameraPopup || refreshCameraPopup) && (
    <Draggable>
    <div className="control-panel action-popup">
      {addCameraPopup && (
        <>
        <button className="close-popup-button" type="button" aria-label="Close popup" aria-disabled="false" onClick={closePopup}>X</button>
        <p> Double click a location on the map and select a camera to add: </p>
        {props.cameras.map((camera) => 
          (isNaN(camera.lon) || isNaN(camera.lat) ? 
          <p onClick={()=>props.addCamera(camera)}> {camera.name} </p> : null)
        )}
        </>
      )}
      {removeCameraPopup && (
        <>
        <button onClick={closePopup}>close</button>
        <button>save</button>
        <p> lets remove that camera </p>
        {props.cameras.map((camera) => 
          <p onClick={()=>props.removeCamera(camera)}> {camera.name} </p>
        )}
        </>
      )}
      {refreshCameraPopup && (
        <>
        <button onClick={closePopup}>close</button>
        <button>save</button>
        <p> lets refresh that camera </p>
        {props.cameras.map((camera) => 
          <p onclick={()=>props.refreshCamera(camera)}> {camera.name} </p>
        )}
        </>
      )}
    </div>
    </Draggable>
    )}
    </div>
  );
}

export default memo(ControlPanel);
