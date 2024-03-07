import { React, useState, memo} from 'react';
import Draggable from 'react-draggable';

function ControlPanel(props) {
  const [addCameraPopup, setAddCameraPopup] = useState(false);
  const [removeCameraPopup, setRemoveCameraPopup] = useState(false);
  const [editCameraPopup, setEditCameraPopup] = useState(false);

  const addCamera = () => {
    props.setControlPanelOpen(true)
    setAddCameraPopup(true)
    setRemoveCameraPopup(false)
    setEditCameraPopup(false)
  }

  const removeCamera = () => {
    props.setControlPanelOpen(false)
    setAddCameraPopup(false)
    setRemoveCameraPopup(true)
    setEditCameraPopup(false)
  }

  const editCamera = () => {
    props.setControlPanelOpen(false)
    setAddCameraPopup(false)
    setRemoveCameraPopup(false)
    setEditCameraPopup(true)
  }

  const closePopup = () => {
    props.setControlPanelOpen(false)
    setAddCameraPopup(false)
    setRemoveCameraPopup(false)
    setEditCameraPopup(false)
  }

  return (
    <div>
    <div className="control-panel mapboxgl-ctrl mapboxgl-ctrl-group-horizontal">
      <button className="add-camera-icon" type="button" aria-label="Add camera" aria-disabled="false" onClick={addCamera}><span className="mapboxgl-ctrl-icon" aria-hidden="true" title="Add camera"></span></button>
      <button className="remove-camera-icon" type="button" aria-label="Remove camera" aria-disabled="false" onClick={removeCamera}><span className="mapboxgl-ctrl-icon" aria-hidden="true" title="Remove camera"></span></button>
      <button className="edit-camera-icon" type="button" aria-label="Edit camera" aria-disabled="false" onClick={editCamera}><span className="mapboxgl-ctrl-icon" aria-hidden="true" title="Edit camera"></span></button>
    </div>
    {(addCameraPopup || removeCameraPopup || editCameraPopup) && (
    <Draggable>
    <div className="control-panel action-popup">
      {addCameraPopup && (
        <>
        <button className="close-popup-button" type="button" aria-label="Close popup" aria-disabled="false" onClick={closePopup}>X</button>
        <p> Double click a location on the map and select a camera to add: </p>
        {props.cameras.map((camera) => 
          (isNaN(camera.lon) || isNaN(camera.lat) || camera.lat == null || camera.lon == null ? 
          <p key={camera.port} className="camera-text" onClick={()=>props.addCamera(camera)}> {camera.name} </p> : null )
        )}
        </>
      )}
      {removeCameraPopup && (
        <>
        <button className="close-popup-button" type="button" aria-label="Close popup" aria-disabled="false" onClick={closePopup}>X</button>
        <p> Select a camera to remove, the settings for the camera won't be removed: </p>
        {props.cameras.map((camera) => 
          (isNaN(camera.lon) || isNaN(camera.lat) || camera.lat == null || camera.lon == null ? 
          null : <p key={camera.port} className="camera-text" onClick={()=>props.removeCamera(camera)}> {camera.name} </p> )
        )}
        </>
      )}
      {editCameraPopup && (
        <>
        <button className="close-popup-button" type="button" aria-label="Close popup" aria-disabled="false" onClick={closePopup}>X</button>
        <p> Select a camera to edit: </p>
        {props.cameras.map((camera) => 
          <p key={camera.port} className="camera-text" onClick={()=>props.editCamera(camera)}> {camera.name} </p>
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
