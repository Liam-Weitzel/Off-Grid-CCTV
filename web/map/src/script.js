import './style.css'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader'

// Textures
const loader = new THREE.TextureLoader()
const height = loader.load('height_map.png')
const texture = loader.load('texture.png')
const under = loader.load('under.jpg')

// Canvas
const canvas = document.querySelector('canvas.webgl')

// Scene
const scene = new THREE.Scene()
new RGBELoader().load(
    'bg.hdr',
    function (texture) {
        texture.mapping = THREE.EquirectangularReflectionMapping
        scene.background = texture
        scene.environment = texture
    }
)

// Objects
const planeGeo = new THREE.PlaneBufferGeometry(3, 3, 64, 64)
const boxGeo = new THREE.BoxGeometry(3, 0.00001, 3, 200, 200, 200)

const pos = boxGeo.attributes.position
const nor = boxGeo.attributes.normal
const enableDisplacement = []
for (let i = 0; i < pos.count; i++) {
  enableDisplacement.push(
    Math.sign(pos.getY(i)),
    Math.sign(nor.getY(i))
  )
  const u = (pos.getX(i) + 3 * 0.5) / 3
  const v = 1 - (pos.getZ(i) + 3 * 0.5) / 3
  boxGeo.attributes.uv.setXY(i, u, v)
}
boxGeo.setAttribute(
  "enableDisp",
  new THREE.Float32BufferAttribute(enableDisplacement, 2)
)

// Materials
const mapMat = new THREE.MeshStandardMaterial({
    color: 'gray',
    map: texture,
    displacementMap: height,
    displacementScale: 2
})

const boxMat = new THREE.MeshStandardMaterial({
    map: under,
    color: 'gray',
    displacementMap: height,
    displacementScale: 2,
    onBeforeCompile: (shader) => {
        shader.vertexShader = `
        attribute vec2 enableDisp;

        ${shader.vertexShader}
        `.replace(
            `#include <displacementmap_vertex>`,
            `
            #ifdef USE_DISPLACEMENTMAP
            if (enableDisp.x > 0.) {

                vec3 vUp = vec3(0, 1, 0);
                vec3 v0 = normalize( vUp ) * ( texture2D( displacementMap, vUv ).x * displacementScale + displacementBias );
                transformed += v0;

                if(enableDisp.y > 0.) {
                    float txl = 1. / 256.;
                    vec3 v1 = normalize( vUp ) * ( texture2D( displacementMap, vUv + vec2(txl, 0.) ).x * displacementScale + displacementBias );
                    v1.xz = vec2(txl, 0.) * 20.;
                    vec3 v2 = normalize( vUp ) * ( texture2D( displacementMap, vUv + vec2(0., txl) ).x * displacementScale + displacementBias );
                    v2.xz = -vec2(0., txl) * 20.;
                    vec3 n = normalize(cross(v1 - v0, v2 - v0));
                    vNormal = normalMatrix * n;
                }              
            }
            #endif
            `
        );
    }
});

// Mesh
const plane = new THREE.Mesh(planeGeo, mapMat)
plane.position.y = 0.01
plane.rotation.x = -Math.PI/2
scene.add(plane)

const box = new THREE.Mesh(boxGeo, boxMat)
scene.add(box)

// Lights
const pointLightBottomLeft = new THREE.PointLight(0xffffff, 1.5)
pointLightBottomLeft.position.x = -10
pointLightBottomLeft.position.y = 10
pointLightBottomLeft.position.z = -10
scene.add(pointLightBottomLeft)

const pointLightTopRight = new THREE.PointLight(0xffffff, 1.5)
pointLightTopRight.position.x = 10
pointLightTopRight.position.y = 10
pointLightTopRight.position.z = 10
scene.add(pointLightTopRight)

/**
 * Sizes
 */
const sizes = {
    width: window.innerWidth,
    height: window.innerHeight
}

window.addEventListener('resize', () =>
{
    // Update sizes
    sizes.width = window.innerWidth
    sizes.height = window.innerHeight

    // Update camera
    camera.aspect = sizes.width / sizes.height
    camera.updateProjectionMatrix()

    // Update renderer
    renderer.setSize(sizes.width, sizes.height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
})

/**
 * Camera
 */
// Base camera
const camera = new THREE.PerspectiveCamera(75, sizes.width / sizes.height, 0.1, 100)
camera.position.x = 0
camera.position.y = 0.8
camera.position.z = 1
scene.add(camera)

/**
 * Controls
 */
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true
controls.maxPolarAngle = Math.PI / 3;
controls.maxDistance = 5;
controls.minDistance = 1.6;

/**
 * Renderer
 */
const renderer = new THREE.WebGLRenderer({
    canvas: canvas
})
renderer.setSize(sizes.width, sizes.height)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

/**
 * Animate
 */

const clock = new THREE.Clock()

const tick = () =>
{

    const elapsedTime = clock.getElapsedTime()

    // Update Orbital Controls
    controls.update()

    // Render
    renderer.render(scene, camera)

    // Call tick again on the next frame
    window.requestAnimationFrame(tick)
}

tick()
