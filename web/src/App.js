import { React, useEffect, useState, useMemo } from 'react';
import JsmpegPlayer from './components/JsmpegPlayer';
import './App.css';
import {createRoot} from 'react-dom/client';
import Map, {Source, Layer, Marker, Popup, NavigationControl, FullscreenControl, ScaleControl, GeolocateControl} from 'react-map-gl';
import Pin from './components/pin';
import ControlPanel from './components/control-panel';
const backendIP = '192.168.92.22';
const apiPort = '8080';
const mapBoxToken = 'pk.eyJ1IjoibGlhbXdlaXR6ZWwiLCJhIjoiNmIwZTUyNWRjMDg5NjVjMTczMTYyOWI2NWZkNmMxZTAifQ.5FiYxafq7rS9Bp1llpWdpw';

const skyLayer = {
  id: 'sky',
  type: 'sky',
  paint: {
    'sky-type': 'atmosphere',
    'sky-atmosphere-sun': [0.0, 0.0],
    'sky-atmosphere-sun-intensity': 15
  }
};

const cameraGeoData = [
  {"name":"camera1","image":"http://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/240px-Above_Gotham.jpg","latitude":36.90453150945084,"longitude":15.013785520105046},
  {"name":"camera2","image":"http://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/240px-Above_Gotham.jpg","latitude":36.90453,"longitude":15.013}
  ]

export default function App() {
  let jsmpegPlayer = null;
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
            // If we let the click event propagates to the map, it will immediately close the popup
            // with `closeOnClick: true`
            e.originalEvent.stopPropagation();
            setPopupInfo(camera);
          }}
        >
          <Pin />
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
      <header className="App-header">
      <input 
        type="text"
        placeholder="search"
        onChange={(e) => searchCameras(e.target.value)}
      /> 

      {cameras.map((camera) => (
        <div key={camera.secret}>
        <p> {camera.secret} </p>
        <JsmpegPlayer
          wrapperClassName="video-wrapper"
          videoUrl={'ws://' + backendIP + ':' + camera.httpPort}
          onRef={ref => jsmpegPlayer = ref}
        />
        </div>
      ))}
      {cameras.length === 0 && <p>No cameras found! </p>} 
      </header>
      <div className="map">
        <Map
          initialViewState={{ 
            latitude: 36.90453150945084,
            longitude: 15.013785520105046,
            zoom: 16,
            bearing: -50,
            pitch: 0
          }}
          maxPitch={85}
          mapStyle="mapbox://styles/mapbox/satellite-v9"
          scrollZoom={false}
          dragPan={false}
          dragRotate={false}
          doubleClickZoom={false}
          touchZoom={false}
          touchRotate={false}
          mapboxAccessToken={mapBoxToken}
          //terrain={{source: 'mapbox-dem', exaggeration: 2.5}}
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
            <Popup
              anchor="top"
              longitude={Number(popupInfo.longitude)}
              latitude={Number(popupInfo.latitude)}
              onClose={() => setPopupInfo(null)}
            >
              <div>
                Display some text about the camera
              </div>
              <img width="100%" src={popupInfo.image} />
            </Popup>
          )}
        </Map>
        <ControlPanel />
      </div>
    </div>
  );
}

export function renderToDom(container) {
  createRoot(container).render(<App />);
}
