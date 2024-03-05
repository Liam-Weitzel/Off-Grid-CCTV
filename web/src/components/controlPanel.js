import { React, useState, memo} from 'react';

function ControlPanel(props) {
  const [addCameraPopUpVisible, setAddCameraPopUpVisible] = useState(false);
  const [removeCameraPopUpVisible, setRemoveCameraPopUpVisible] = useState(false);
  const [refreshCameraPopUpVisible, setRefreshCameraPopUpVisible] = useState(false);
  
  const addCamera = () => {
    setAddCameraPopUpVisible(true);
    setRefreshCameraPopUpVisible(false);
    setRemoveCameraPopUpVisible(false);
  }

  const removeCamera = () => {
    setAddCameraPopUpVisible(false);
    setRefreshCameraPopUpVisible(false);
    setRemoveCameraPopUpVisible(true);
  }

  const refreshCamera = () => {
    setAddCameraPopUpVisible(false);
    setRefreshCameraPopUpVisible(true);
    setRemoveCameraPopUpVisible(false);
  }

  return (
    <div>
    <div className="control-panel mapboxgl-ctrl mapboxgl-ctrl-group-horizontal">
      <button className="add-camera-icon" type="button" aria-label="Add camera" aria-disabled="false" onClick={addCamera}><span className="mapboxgl-ctrl-icon" aria-hidden="true" title="Add camera"></span></button>
      <button className="remove-camera-icon" type="button" aria-label="Remove camera" aria-disabled="false" onClick={removeCamera}><span className="mapboxgl-ctrl-icon" aria-hidden="true" title="Remove camera"></span></button>
      <button className="refresh-camera-icon" type="button" aria-label="Refresh camera" aria-disabled="false" onClick={refreshCamera}><span className="mapboxgl-ctrl-icon" aria-hidden="true" title="Refresh camera"></span></button>
    </div>
    {(addCameraPopUpVisible || removeCameraPopUpVisible || refreshCameraPopUpVisible) && (
    <div className="control-panel action-popup">
      {addCameraPopUpVisible && (
        <p> lets add that camera </p>
      )}
      {removeCameraPopUpVisible && (
        <p> lets remove that camera </p>
      )}
      {refreshCameraPopUpVisible && (
        <p> lets refresh that camera </p>
      )}
    </div>
    )}
    </div>
  );
}

export default memo(ControlPanel);
