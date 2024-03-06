import { React, useState, useEffect } from 'react';
import JsmpegPlayer from './components/jsmpegPlayer';
import './App.css';
import { createRoot } from 'react-dom/client';
import Map, { Source, Layer, Marker, Popup, NavigationControl, FullscreenControl } from 'react-map-gl';
import CameraSvg from './components/cameraSvg';
import TempCameraSvg from './components/tempCameraSvg'
import FullscreenSvg from './components/fullscreenSvg';
import ControlPanel from './components/controlPanel';

const backendIP = '192.168.92.22';
const apiPort = '8080';
const mapBoxToken = 'pk.eyJ1IjoibGlhbXdlaXR6ZWwiLCJhIjoiNmIwZTUyNWRjMDg5NjVjMTczMTYyOWI2NWZkNmMxZTAifQ.5FiYxafq7rS9Bp1llpWdpw';

const response = await fetch (`http://${backendIP}:${apiPort}/fetchConfigs`);
const configs = await response.json();

const initialViewState = { 
  latitude: configs.latitude,
  longitude: configs.longitude,
  zoom: configs.zoom,
  bearing: configs.bearing,
  pitch: configs.pitch
};

const skyLayer = {
  id: configs.id,
  type: configs.type,
  paint: configs.paint
};

export default function App() {
  const [cameras, setCameras] = useState([]);
  const [popupInfo, setPopupInfo] = useState(null);
  const [pins, setPins] = useState(null);
  const [lastClickedPos, setLastClickedPos] = useState([]);
  const [controlPanelOpen, setControlPanelOpen] = useState(false);

  useEffect(() => {
    return () => { //this runs once on app start
      fetchCameras();
    };
  }, []);

  const handleClick = (e) => {
    const [lon, lat] = e.lngLat.toArray();
    if(controlPanelOpen) setLastClickedPos({ lon, lat });
  };

  const fetchCameras = async () => {
    const response = await fetch (`http://${backendIP}:${apiPort}/fetchCameras`);
    const data = await response.json();
    setCameras(data);
    setPins(data.map((camera, index) => (
      ( !isNaN(camera.lon) && !isNaN(camera.lat) && camera.lat != null && camera.lon != null && (
      <Marker
      key={`marker-${index}`}
      longitude={camera.lon}
      latitude={camera.lat}
      anchor="bottom"
      onClick={e => {
        e.originalEvent.stopPropagation();
        setPopupInfo({name: camera.name, lon: camera.lon, lat: camera.lat, httpPort: camera.httpPort});
      }}
      >
      <CameraSvg />
      </Marker>))
    )));
  };

  useEffect(() => { //Remove temp camera marker when controlPanel closes
    if(!controlPanelOpen && pins != null ) {
      setLastClickedPos([]);
      let pinsCopy = [...pins];
      pinsCopy.forEach(function(pin, index, object){
        if(pin.key === 'temp') {
          object.splice(index, 1);
        }
      });
      setPins(pinsCopy);
    }
  }, [controlPanelOpen]);

  useEffect(() => { //Add temp camera marker when controlPanel is open
    ( !isNaN(lastClickedPos.lon) && !isNaN(lastClickedPos.lat) && pins != null && controlPanelOpen && (
    setPins(pins => [...pins.slice(0, -1), (
      <Marker
      key={'temp'}
      longitude={lastClickedPos.lon.toString()}
      latitude={lastClickedPos.lat.toString()}
      >
      <TempCameraSvg />
      </Marker>)])))
  }, [lastClickedPos, controlPanelOpen]);

  const addCamera = async (camera) => {

  }

  const removeCamera = async (camera) => {

  }

  const refreshCamera = async (camera) => {

  }

  return (
    <div className="App">
    <Map
    initialViewState={initialViewState}
    maxPitch={60}
    mapStyle="mapbox://styles/mapbox/satellite-v9"
    scrollZoom={false}
    dragPan={false}
    dragRotate={true}
    doubleClickZoom={false}
    touchZoom={false}
    touchRotate={true}
    mapboxAccessToken={mapBoxToken}
    terrain={{source: 'mapbox-dem', exaggeration: 2.5}}
    onDblClick={handleClick}
    >
    <Source
    id="mapbox-dem"
    type="raster-dem"
    url="mapbox://mapbox.mapbox-terrain-dem-v1"
    tileSize={512}
    maxzoom={16}
    />
    <Layer {...skyLayer} />
    <FullscreenControl position="top-left" />
    <NavigationControl position="top-left" />

    {pins}

    {popupInfo && (
      <Popup key={popupInfo.name}
      className = "marker-popup"
      anchor="top"
      longitude={Number(popupInfo.lon)}
      latitude={Number(popupInfo.lat)}
      onClose={() => {
        setPopupInfo(null);
      }}
      >
      <JsmpegPlayer
      wrapperClassName="video-wrapper"
      videoUrl={'ws://' + backendIP + ':' + popupInfo.httpPort}
      />
      <FullscreenSvg />
      </Popup>
    )}

    <ControlPanel 
      lastClickedPos={lastClickedPos}
      setControlPanelOpen={setControlPanelOpen}
      cameras={cameras}
      addCamera={addCamera}
      removeCamera={removeCamera}
      refreshCamera={refreshCamera}
    />
    </Map>
    </div>
  );
}

export function renderToDom(container) {
  createRoot(container).render(<App />);
}
