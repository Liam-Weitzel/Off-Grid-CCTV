import * as React from 'react';

const ICON = `M18.15,4.94A2.09,2.09,0,0,0,17,5.2l-8.65,5a2,2,0,0,0-.73,2.74l1.5,2.59a2,2,0,0,0,2.73.74l1.8-1a2.49,2.49,0,0,0,1.16,1V18a2,2,0,0,0,2,2H22V18H16.81V16.27A2.49,2.49,0,0,0,18,12.73l2.53-1.46a2,2,0,0,0,.74-2.74l-1.5-2.59a2,2,0,0,0-1.59-1M6.22,13.17,2,13.87l.75,1.3,2,3.46.75,1.3,2.72-3.3Z`;

const SVGStyle = {
  cursor: 'pointer',
  fill: '#000000',
  stroke: '#ffffff',
  opacity:'0.6'
};

function TempCameraSvg({size = 50}) {
  return (
    <svg height={size} viewBox="0 0 24 24" style={SVGStyle}>
      <path d={ICON} />
    </svg>
  );
}

export default React.memo(TempCameraSvg);
