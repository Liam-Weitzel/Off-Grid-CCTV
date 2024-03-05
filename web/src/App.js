import { React, useState, useEffect, useMemo } from 'react';
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

  useEffect(() => {
    return () => { //this runs once on app start
      fetchCameras();
    };
  }, []);

  const fetchCameras = async () => {
    const response = await fetch (`http://${backendIP}:${apiPort}/fetchCameras`);
    const data = await response.json();
    setCameras(data);
    setPins(data.map((camera, index) => (
      <Marker
      key={`marker-${index}`}
      longitude={parseFloat(camera.lon)}
      latitude={parseFloat(camera.lat)}
      anchor="bottom"
      onClick={e => {
        e.originalEvent.stopPropagation();
        setPopupInfo(camera);
      }}
      >
      <CameraSvg />
      </Marker>
    )));
  };

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

    <ControlPanel />
    </Map>
    </div>
  );
}

export function renderToDom(container) {
  createRoot(container).render(<App />);
}
