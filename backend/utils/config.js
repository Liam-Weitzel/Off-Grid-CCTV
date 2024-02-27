/**
 * Here to set opencv configs
 */
exports.opencv = {
	// set webcam port
	camPort: 1,
	// set webcam FPS
	camFps: 30,
	// set frame size
	frameSize: 640
};

/**
 * Here to set server configs
 */
exports.server = {
	// set http port
	httpPort: 8080,
	// set websocket port
	wsPort: 8081,
	// set stream port
	streamPort: 8082,
	// set stream secret
	streamSecret: 'N23y08VnzfDH4Wmf2tXoDyxbwf2rGQJC'
};

/**
 * Here to set tensorflow object detection class names
 */
exports.classNames = {
	0: 'person',
	1: 'bicycle',
	2: 'car',
	3: 'motorcycle',
	4: 'truck',
	5: 'cat',
	6: 'dog',
	7: 'horse',
	8: 'sheep',
	9: 'cow',
	10: 'bear',
	11: 'zebra',
};
