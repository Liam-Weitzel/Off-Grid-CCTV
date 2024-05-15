import './style.css'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader'

const scene = new THREE.Scene()
const sizes = {
    width: window.innerWidth,
    height: window.innerHeight
}

const loader = new THREE.TextureLoader()
const height_map = loader.load('height_map.png')
const texture = loader.load('texture.png')
const under = loader.load('under.jpg')

new RGBELoader().load('bg.hdr', function (texture) {
    texture.mapping = THREE.EquirectangularReflectionMapping
    scene.background = texture
    scene.environment = texture
})

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000)
camera.position.set(0, 8, 8)

const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true })
renderer.setSize(sizes.width, sizes.height)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
document.body.appendChild(renderer.domElement)

const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true
controls.maxPolarAngle = Math.PI / 2.1
controls.maxDistance = 200
controls.minDistance = 1
controls.target.set(0, 0, 0)

// Objects
const planeGeo = new THREE.PlaneBufferGeometry(3, 3, 100, 100).rotateX(-Math.PI * 0.5)
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
    map: texture,
})

const boxMat = new THREE.MeshStandardMaterial({
    map: under,
    displacementMap: height_map,
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
scene.add(plane)

const box = new THREE.Mesh(boxGeo, [boxMat,boxMat,null,null,boxMat,boxMat])
scene.add(box)

loader.load('height_map.png', function (t) {
  const canvas = document.createElement("canvas")
  canvas.width = t.image.width
  canvas.height = t.image.height
  const ctx = canvas.getContext("2d")
  ctx.drawImage(t.image, 0, 0, t.image.width, t.image.height)

  const wdth = planeGeo.parameters.widthSegments + 1
  const hght = planeGeo.parameters.heightSegments + 1
  const widthStep = t.image.width / wdth
  const heightStep = t.image.height / hght
  console.log(wdth, hght, widthStep, heightStep)

  for (let h = 0; h < hght; h++) {
    for (let w = 0; w < wdth; w++) {
      const imgData = ctx.getImageData(Math.round(w * widthStep), Math.round(h * heightStep), 1, 1).data
      let displacementVal = imgData[0] / 255.0
      displacementVal *= 2
      const idx = (h * wdth) + w
      planeGeo.attributes.position.setY(idx, displacementVal)
    }
  }
  planeGeo.attributes.position.needsUpdate = true
  planeGeo.computeVertexNormals()
})

const points = []
points.push(new THREE.Vector3(-90, 0, 0))
points.push(new THREE.Vector3(90, 0, 0))
const latLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: 0x00ff00 })
)
scene.add(latLine)

const pointLightBottomLeft = new THREE.PointLight(0xffffff, 0.9)
pointLightBottomLeft.position.x = -10
pointLightBottomLeft.position.y = 10
pointLightBottomLeft.position.z = -10
scene.add(pointLightBottomLeft)

const pointLightTopRight = new THREE.PointLight(0xffffff, 0.9)
pointLightTopRight.position.x = 10
pointLightTopRight.position.y = 10
pointLightTopRight.position.z = 10
scene.add(pointLightTopRight)

const lonLine = latLine.clone()
lonLine.rotateY(Math.PI / 2)
scene.add(lonLine)

const altLine = latLine.clone()
altLine.rotateZ(Math.PI / 2)
scene.add(altLine)

const mouse = new THREE.Vector2()
const raycaster = new THREE.Raycaster()

function onDoubleClick(event) {
    mouse.set(
        (event.clientX / renderer.domElement.clientWidth) * 2 - 1,
        -(event.clientY / renderer.domElement.clientHeight) * 2 + 1
    )
    raycaster.setFromCamera(mouse, camera)
    const intersects = raycaster.intersectObject(plane, false)
    if (intersects.length > 0) {
        const { point, uv } = intersects[0]
        console.log(point)
    }
}
renderer.domElement.addEventListener('dblclick', onDoubleClick, false)

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
