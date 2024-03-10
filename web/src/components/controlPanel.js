import { React, useState, useEffect, memo} from 'react';
import Draggable from 'react-draggable';

function ControlPanel(props) {
  const [addCameraPopup, setAddCameraPopup] = useState(false);
  const [removeCameraPopup, setRemoveCameraPopup] = useState(false);
  const [editCameraPopup, setEditCameraPopup] = useState(false);
  const [editCameraPopupChild, setEditCameraPopupChild] = useState(false);

  const addCamera = () => {
    props.setControlPanelOpen(true);
    setAddCameraPopup(true);
    setRemoveCameraPopup(false);
    setEditCameraPopup(false);
    setEditCameraPopupChild(false);
  }

  const removeCamera = () => {
    props.setControlPanelOpen(false);
    setAddCameraPopup(false);
    setRemoveCameraPopup(true);
    setEditCameraPopup(false);
    setEditCameraPopupChild(false);
  }

  const editCamera = () => {
    props.setControlPanelOpen(false);
    setAddCameraPopup(false);
    setRemoveCameraPopup(false);
    setEditCameraPopup(true);
    setEditCameraPopupChild(false);
  }

  const closePopup = () => {
    props.setControlPanelOpen(false);
    setAddCameraPopup(false);
    setRemoveCameraPopup(false);
    setEditCameraPopup(false);
    setEditCameraPopupChild(false);
  }

  useEffect(() => {
    if(props.controlPanelOpen == false && editCameraPopupChild != false) closePopup();
  }, [props.controlPanelOpen]);

  const updateCameraAttribute = (e) => {
    if(e.target.value != editCameraPopupChild[e.target.id]) {
      editCameraPopupChild[e.target.id] = e.target.value;
      setEditCameraPopupChild({...editCameraPopupChild});
    }
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
        {!editCameraPopupChild ? 
          <>
          <p> Select a camera to edit </p>
          {props.cameras.map((camera) => 
            <p 
            key={camera.port}
            className="camera-text"
            onClick={()=>{
              setEditCameraPopupChild(camera);
              props.setControlPanelOpen(true)
            }}> {camera.name} </p>
          )} 
          </>
          : 
          <> 
          <p> Editing <a className="important">{editCameraPopupChild.name}</a>. Double click on the map to edit its location. <a className="important">Some settings require a server restart to take effect.</a></p>
          {Object.values(editCameraPopupChild).map((val,key) => (
            <>
            {["port", "path", "lon", "lat"].includes(Object.keys(editCameraPopupChild)[key]) ? null :
              <tr>
              <td>
              <p> {Object.keys(editCameraPopupChild)[key]}: </p>
              </td> 
              <td>
              <input 
              id={Object.keys(editCameraPopupChild)[key]}
              type="text" 
              defaultValue={val}
              onChange={e => updateCameraAttribute(e)}
              />
              </td>
              </tr> 
            }
            </>
          ))} 
          <button className="edit-camera-save-button" onClick={()=>props.editCamera(editCameraPopupChild)}>save</button>
          </>
        }
        </>
      )}
      </div>
      </Draggable>
    )}
    </div>
  );
}

export default memo(ControlPanel);
