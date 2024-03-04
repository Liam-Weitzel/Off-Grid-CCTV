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
    <div className="control-panel">
      <button onClick={addCamera}> + </button>
      <button onClick={removeCamera}> - </button>
      <button onClick={refreshCamera}> r </button>
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
