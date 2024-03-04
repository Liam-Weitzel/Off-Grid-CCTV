import { React, useState, useMemo } from 'react';
import JsmpegPlayer from './components/jsmpegPlayer';
import './App.css';
import { createRoot } from 'react-dom/client';
import Map, { Source, Layer, Marker, Popup, NavigationControl, FullscreenControl } from 'react-map-gl';
import CameraSvg from './components/cameraSvg.js';
import FullscreenSvg from './components/fullscreenSvg.js';
import ControlPanel from './components/controlPanel';

const backendIP = '192.168.92.22';
const apiPort = '8080';
const mapBoxToken = 'pk.eyJ1IjoibGlhbXdlaXR6ZWwiLCJhIjoiNmIwZTUyNWRjMDg5NjVjMTczMTYyOWI2NWZkNmMxZTAifQ.5FiYxafq7rS9Bp1llpWdpw';

const initialViewState = { 
  latitude: 36.90453150945084,
  longitude: 15.013785520105046,
  zoom: 16,
  bearing: -50,
  pitch: 0
};

const cameraGeoData = [
  {"key":"camera1","image":"http://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/240px-Above_Gotham.jpg","latitude":36.90453150945084,"longitude":15.013785520105046, "httpPort":8084},
  {"key":"camera2","image":"http://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/240px-Above_Gotham.jpg","latitude":36.90353150945084,"longitude":15.013185520105046, "httpPort":8081}
];

const skyLayer = {
  id: 'sky',
  type: 'sky',
  paint: {
    'sky-type': 'atmosphere',
    'sky-atmosphere-sun': [0.0, 0.0],
    'sky-atmosphere-sun-intensity': 15
  }
};

export default function App() {
  const [cameras, setCameras] = useState([]);
  const [popupInfo, setPopupInfo] = useState(null);

  const pins = useMemo(
    () =>
    cameraGeoData.map((camera, index) => (
      <Marker
      key={`marker-${index}`}
      longitude={camera.longitude}
      latitude={camera.latitude}
      anchor="bottom"
      onClick={e => {
        e.originalEvent.stopPropagation();
        setPopupInfo(camera);
      }}
      >
      <CameraSvg />
      </Marker>
    )),
    []
  );

  const searchCameras = async (q) => {
    const response = await fetch (`http://${backendIP}:${apiPort}/searchCameras?` + new URLSearchParams({q}));
    const data = await response.json();
    setCameras(data);
  };

  return (
    <div className="App">
    {/* <header className="App-header">
      <input 
      type="text"
      placeholder="search"
      onChange={(e) => searchCameras(e.target.value)}
      /> 

      {cameras.map((camera) => (
        <JsmpegPlayer
        wrapperClassName="video-wrapper"
        videoUrl={'ws://' + backendIP + ':' + 8081}
        onRef={ref => jsmpegPlayer = ref}
        />
      ))}
      {cameras.length === 0 && <p>No cameras found! </p>} 
      </header> */}
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
      <Popup key={popupInfo.key}
      className = "marker-popup"
      anchor="top"
      longitude={Number(popupInfo.longitude)}
      latitude={Number(popupInfo.latitude)}
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

    <ControlPanel />
    </Map>
    </div>
  );
}

export function renderToDom(container) {
  createRoot(container).render(<App />);
}
