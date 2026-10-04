// Bake the complete CC0 scan once: decimating millions of tiny needles removes
// canopy coverage. A four-view atlas keeps that coverage at a mobile-sized cost.
import {createServer} from 'node:http';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import sharp from 'sharp';

const raw=path.resolve(process.argv[2]),root=process.cwd();
const html=`<!doctype html><script type="importmap">{"imports":{"three":"/three/build/three.module.js","three/addons/":"/three/examples/jsm/"}}</script><script type="module">
import * as T from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const renderer=new T.WebGLRenderer({alpha:true,antialias:true,preserveDrawingBuffer:true});
renderer.setSize(1024,2048);renderer.setClearColor(0,0);renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=.88;
document.body.append(renderer.domElement);
const loaded=await new GLTFLoader().loadAsync('/raw/fir/selected.gltf');
const scene=new T.Scene(),tree=loaded.scene;scene.add(tree);
const alpha=await new T.TextureLoader().loadAsync('/raw/fir/twig-alpha.png');alpha.flipY=false;
tree.traverse(o=>{if(o.isMesh){o.material.roughness=.95;o.material.metalness=0;if(o.material.name.includes('twig')){o.material.alphaMap=alpha;o.material.transparent=false;o.material.alphaTest=.12;o.material.side=T.DoubleSide;}}});
const box=new T.Box3().setFromObject(tree),center=box.getCenter(new T.Vector3());
tree.position.set(-center.x,-box.min.y,-center.z);const scale=14.4/(box.max.y-box.min.y);tree.scale.setScalar(scale);tree.position.multiplyScalar(scale);
scene.add(new T.HemisphereLight('#d7e8ed','#59643a',2));scene.add(new T.AmbientLight('#e1ecd3',.35));
const sun=new T.DirectionalLight('#fff0d6',2.8);sun.position.set(-12,24,10);scene.add(sun);
const camera=new T.OrthographicCamera(-4,4,7.65,-7.65,.1,100);camera.position.set(0,7.45,30);camera.lookAt(0,7.45,0);
window.bake=angle=>{tree.rotation.y=angle;renderer.render(scene,camera);return renderer.domElement.toDataURL('image/png');};
window.ready=true;
</script>`;
const server=createServer(async(req,res)=>{try{
 if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end(html);return;}
 const prefix=req.url.startsWith('/raw/')?'/raw/':'/three/',base=prefix==='/raw/'?raw:path.join(root,'node_modules/three');
 const file=path.resolve(base,decodeURIComponent(req.url.slice(prefix.length)));if(!file.startsWith(base+path.sep))throw Error('path');
 res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':file.endsWith('.gltf')?'application/json':'application/octet-stream');res.end(await readFile(file));
 }catch{res.statusCode=404;res.end();}});
await new Promise(r=>server.listen(5188,'127.0.0.1',r));
const browser=await chromium.launch({channel:'chrome',headless:true});
try{
 const page=await browser.newPage();page.on('pageerror',console.error);await page.goto('http://127.0.0.1:5188');
 await page.waitForFunction(()=>window.ready,{},{timeout:180000});
 const panels=[];
 for(let i=0;i<4;i++){
  const data=await page.evaluate(a=>window.bake(a),i*Math.PI/2),png=Buffer.from(data.split(',')[1],'base64');
  panels.push({input:await sharp(png).resize(512,1024).png().toBuffer(),left:(i%2)*512,top:Math.floor(i/2)*1024});
 }
 const out=path.join(root,'public/courses/grenland/myhra/fir-atlas.webp');
 await sharp({create:{width:1024,height:2048,channels:4,background:{r:0,g:0,b:0,alpha:0}}}).composite(panels).webp({quality:94,alphaQuality:100}).toFile(out);
 console.log(out);
 await writeFile(path.join(raw,'fir/atlas-provenance.json'),JSON.stringify({source:'https://polyhaven.com/a/fir_tree_01',license:'CC0',method:'Four views of the full source canopy rendered with Three.js; no geometry decimation.'},null,2));
}finally{await browser.close();server.close();}
