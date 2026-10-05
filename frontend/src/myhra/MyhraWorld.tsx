import {Suspense,memo,useEffect,useMemo,useRef} from 'react';
import {useFrame,useThree,type ThreeEvent} from '@react-three/fiber';
import {useGLTF,useTexture} from '@react-three/drei';
import {AmbientLight,Box3,BufferGeometry,Color,DirectionalLight,DoubleSide,Float32BufferAttribute,HemisphereLight,InstancedMesh,Mesh,MeshBasicMaterial,MeshStandardMaterial,Object3D,OrthographicCamera,PlaneGeometry,RedFormat,RepeatWrapping,Scene,SRGBColorSpace,Vector2,Vector3,WebGLRenderTarget} from 'three';
import {terrainGeometry} from '../course/terrain';
import {contains,heightAt,surfaceAt,type Vec3} from '../course/engine';
import {ASSETS,type Hole} from './data';

const random=(seed:number)=>()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
export const GROUND_TEXTURES=['surface-mask.png','grass_ground-diff.webp','path-mask.png','sand_02-diff.webp','grass_ground-nor_gl.webp','gravel_road-diff.webp','sand_02-nor_gl.webp','gravel_road-nor_gl.webp'].map(file=>ASSETS+file);

export const Ground=memo(function Ground({hole,onAim,anchor,cameraInput}:{hole:Hole;onAim:(p:Vec3)=>void;anchor:Vec3;cameraInput?:{readonly current:{readonly gesture:string}}}){
 const [mask,detail,path,sand,normal,gravel,sandNormal,gravelNormal]=useTexture(GROUND_TEXTURES,textures=>{
  for(const i of [1,3,4,5,6,7]){textures[i].wrapS=textures[i].wrapT=RepeatWrapping;textures[i].anisotropy=8;}
  for(const i of [1,3,5])textures[i].colorSpace=SRGBColorSpace;textures[4].repeat.set(768*.5,768*.5);textures[2].format=RedFormat;textures[2].needsUpdate=true;
 });
  const geometry=useMemo(()=>{
   const tile=hole.world.terrain,n=tile.size,g=terrainGeometry(tile,hole.world.detail,768),indices:number[]=[],detail=hole.world.detail;
   const near=(x:number,z:number)=>{const wx=x*tile.spacing,wz=z*tile.spacing;return (wx>=300&&wx<=466&&wz>=180&&wz<=650)||Math.hypot(wx-anchor[0],wz-anchor[2])<42||(!!detail&&wx>detail.origin[0]-8&&wx<detail.origin[0]+(detail.size-1)*detail.spacing+8&&wz>detail.origin[1]-8&&wz<detail.origin[1]+(detail.size-1)*detail.spacing+8);};
   // The entire Myhra playing corridor retains every authoritative triangle.
   // Only surrounding distant scenery is coarsened, identically in all tiers.
   for(let z=0;z<n-4;z+=4)for(let x=0;x<n-4;x+=4){
    const i=z*n+x;if(detail&&contains(detail,x*tile.spacing,z*tile.spacing)&&contains(detail,(x+4)*tile.spacing,(z+4)*tile.spacing))continue;
    if(near(x,z)||near(x+4,z+4)){for(let dz=0;dz<4;dz++)for(let dx=0;dx<4;dx++){const a=i+dz*n+dx,wx=(x+dx)*tile.spacing,wz=(z+dz)*tile.spacing;if(detail&&contains(detail,wx,wz)&&contains(detail,wx+tile.spacing,wz+tile.spacing))continue;indices.push(a,a+n,a+1,a+1,a+n,a+n+1);}}
    else if(near(x-4,z)||near(x+4,z)||near(x,z-4)||near(x,z+4)){
     // Stitch the coarse cell to its fine neighbours at the shared midpoints.
     const edge=[i,i+n,i+n*2,i+n*3,i+n*4,i+n*4+1,i+n*4+2,i+n*4+3,i+n*4+4,i+n*3+4,i+n*2+4,i+n+4,i+4,i+3,i+2,i+1];for(let j=0;j<16;j++)indices.push(i+n*2+2,edge[j],edge[(j+1)%16]);
    }else indices.push(i,i+n*4,i+4,i+4,i+n*4,i+n*4+4);
   }g.setIndex(indices);g.computeVertexNormals();return g;
  },[hole,anchor]);
 const green=useMemo(()=>terrainGeometry(hole.world.detail!,undefined,768),[hole]);
 const material=useMemo(()=>{
  const m=new MeshStandardMaterial({roughness:.97,color:'#ffffff',normalMap:normal,normalScale:new Vector2(.28,.28)});
  m.onBeforeCompile=s=>{
   s.uniforms.surfaceMap={value:mask};s.uniforms.detailMap={value:detail};s.uniforms.pathMap={value:path};s.uniforms.sandMap={value:sand};s.uniforms.gravelMap={value:gravel};s.uniforms.sandNormal={value:sandNormal};s.uniforms.gravelNormal={value:gravelNormal};
   s.vertexShader='varying vec3 vField;\n'+s.vertexShader;s.vertexShader=s.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\nvField=position;');
   s.fragmentShader=`varying vec3 vField; uniform sampler2D surfaceMap,detailMap,pathMap,sandMap,gravelMap,sandNormal,gravelNormal;
    float fieldHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    float fieldNoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(fieldHash(i),fieldHash(i+vec2(1,0)),f.x),mix(fieldHash(i+vec2(0,1)),fieldHash(i+vec2(1,1)),f.x),f.y);}
   `+s.fragmentShader;
   s.fragmentShader=s.fragmentShader.replace('#include <map_fragment>',`
    vec4 field=texture2D(surfaceMap,vec2(vField.x/768.,1.-vField.z/768.));
    float noise=fieldNoise(vField.xz*.065)*.6+fieldNoise(vField.xz*.31)*.3+fieldNoise(vField.xz*4.)*.1;
    float stripe=smoothstep(-.25,.25,sin((vField.z+vField.x*.13)*.48));
    vec3 rough=mix(vec3(.07,.118,.024),vec3(.18,.23,.06),noise);
    vec3 fairway=mix(vec3(.063,.144,.027),vec3(.092,.176,.043),stripe)+noise*.014;
    vec3 putting=mix(vec3(.165,.268,.083),vec3(.188,.291,.100),stripe)+noise*.01;
    vec3 sand=texture2D(sandMap,vField.xz*.45).rgb*vec3(1.28,1.22,1.10);
    sand*=.96+.04*sin(vField.x*18.+sin(vField.z*.8)*2.);
    vec3 surface=mix(rough,fairway,field.r);surface=mix(surface,putting,field.g);surface=mix(surface,sand,field.b);
    vec3 fine=texture2D(detailMap,vField.xz*.5).rgb;
    float grains=dot(fine,vec3(.333));surface*=mix(.53+grains*2.7,.91+grains*.4,field.g);
    surface*=1.-field.a*.14;
    float road=texture2D(pathMap,vec2(vField.x/768.,1.-vField.z/768.)).r;
    vec3 gravel=texture2D(gravelMap,vField.xz*.45).rgb*.72;surface=mix(surface,gravel,road);
    diffuseColor.rgb*=surface;
   `);
   s.fragmentShader=s.fragmentShader.replace('#include <normal_fragment_maps>',`
    vec3 mapN=texture2D(normalMap,vNormalMapUv).xyz*2.-1.;mapN.xy*=mix(.75,.18,max(field.r,field.g));
    vec3 sandN=texture2D(sandNormal,vec2(vField.x,-vField.z)*.45).xyz*2.-1.;sandN.xy*=.45;
    vec3 roadN=texture2D(gravelNormal,vec2(vField.x,-vField.z)*.45).xyz*2.-1.;roadN.xy*=.65;
    mapN=mix(mix(mapN,sandN,field.b),roadN,road);normal=normalize(tbn*mapN);
   `);
  };m.customProgramCacheKey=()=> 'myhra-field-photo-01';return m;
 },[mask,detail,path,sand,normal,gravel,sandNormal,gravelNormal]);
  const down=useRef(new Map<number,[number,number]>()),multi=useRef(false),lastAim=useRef(0);
 const start=(e:ThreeEvent<PointerEvent>)=>{if(e.button!==0)return;e.stopPropagation();down.current.set(e.pointerId,[e.clientX,e.clientY]);if(down.current.size>1)multi.current=true;};
  const end=(e:ThreeEvent<PointerEvent>)=>{const origin=down.current.get(e.pointerId);if(e.button===0&&origin&&!multi.current&&(!cameraInput||cameraInput.current.gesture==='aim')&&Math.hypot(e.clientX-origin[0],e.clientY-origin[1])<6){e.stopPropagation();onAim(e.point.toArray() as Vec3);}};
  const move=(e:ThreeEvent<PointerEvent>)=>{const origin=down.current.get(e.pointerId);if(origin&&!multi.current&&(!cameraInput||cameraInput.current.gesture==='aim')&&down.current.size===1&&Math.hypot(e.clientX-origin[0],e.clientY-origin[1])>=6&&performance.now()-lastAim.current>40){lastAim.current=performance.now();onAim(e.point.toArray() as Vec3);}};
 useEffect(()=>{const clear=(e:PointerEvent)=>{down.current.delete(e.pointerId);if(!down.current.size)multi.current=false;},reset=()=>{down.current.clear();multi.current=false;};addEventListener('pointerup',clear);addEventListener('pointercancel',clear);addEventListener('blur',reset);addEventListener('orientationchange',reset);document.addEventListener('visibilitychange',reset);return()=>{removeEventListener('pointerup',clear);removeEventListener('pointercancel',clear);removeEventListener('blur',reset);removeEventListener('orientationchange',reset);document.removeEventListener('visibilitychange',reset);};},[]);
 useEffect(()=>()=>geometry.dispose(),[geometry]);
 useEffect(()=>()=>green.dispose(),[green]);
 useEffect(()=>()=>material.dispose(),[material]);
  return <group><mesh geometry={geometry} material={material} receiveShadow onPointerDown={start} onPointerMove={move} onPointerUp={end}/><mesh geometry={green} material={material} receiveShadow onPointerDown={start} onPointerMove={move} onPointerUp={end}/></group>;
});

function TreeLayer({hole,anchor,high,birch=false}:{hole:Hole;anchor:Vec3;high:boolean;birch?:boolean}){
 const {scene}=useGLTF(ASSETS+(birch?(high?'birch.glb':'birch-mobile.glb'):'fir.glb'));const {gl}=useThree();
 const target=useMemo(()=>{const t=new WebGLRenderTarget(512,1024);t.texture.colorSpace=SRGBColorSpace;return t;},[]),impostor=target.texture;
 const near=useRef<(InstancedMesh|null)[]>([]),far=useRef<InstancedMesh>(null);
 const source=useMemo(()=>{
  const parts:Mesh[]=[];scene.updateMatrixWorld(true);const box=new Box3().setFromObject(scene),scale=14.4/(box.max.y-box.min.y),center=box.getCenter(new Vector3());scene.traverse(o=>{if(o instanceof Mesh){const m=o.clone();m.geometry=o.geometry.clone().applyMatrix4(o.matrixWorld).scale(scale,scale,scale).translate(-center.x*scale,-box.min.y*scale,-center.z*scale);m.position.set(0,0,0);m.rotation.set(0,0,0);m.scale.setScalar(1);m.material=(o.material as MeshStandardMaterial).clone();const material=m.material as MeshStandardMaterial;material.envMapIntensity=.35;material.roughness=.9;material.metalness=0;parts.push(m);}});return parts;
 },[scene]);
 useEffect(()=>{
  const stage=new Scene(),tree=new Object3D();source.forEach(p=>tree.add(p.clone()));stage.add(tree);
  stage.add(new HemisphereLight('#d7e8ed','#4b5d30',2.0));stage.add(new AmbientLight('#dfe8cf',.35));
  const sun=new DirectionalLight('#fff0d6',3);sun.position.set(-12,24,10);stage.add(sun);
  const box=new Box3().setFromObject(tree),center=box.getCenter(new Vector3());
  const camera=new OrthographicCamera(-4,4,7.65,-7.65,.1,100);camera.position.set(center.x,7.45,30);camera.lookAt(center.x,7.45,0);
  const oldTarget=gl.getRenderTarget(),oldColor=gl.getClearColor(new Color()),oldAlpha=gl.getClearAlpha();
  gl.setRenderTarget(target);gl.setClearColor(0x000000,0);gl.clear();gl.render(stage,camera);gl.setRenderTarget(oldTarget);gl.setClearColor(oldColor,oldAlpha);
  return()=>target.dispose();
 },[source,gl,target]);
 const placements=useMemo(()=>{
  const sorted=hole.data.trees.filter((_,i)=>birch?i%3===0:i%3!==0).map(t=>({t:birch?[t[0],t[1],t[2],t[3]*.72,t[4]]:t,d:Math.hypot(t[0]-anchor[0],t[2]-anchor[2])})).filter(o=>o.d<600).sort((a,b)=>a.d-b.d);
   const count=birch?(high?8:3):(high?4:2),close=sorted.filter(o=>o.d<85).slice(0,count);return {near:close.map(o=>o.t),far:sorted.slice(close.length).map(o=>o.t)};
 },[hole,anchor,high,birch]);
 const plane=useMemo(()=>new PlaneGeometry(8,15.3).translate(0,7.55,0),[]);
 const billboard=useMemo(()=>{
  const m=new MeshBasicMaterial({map:impostor,alphaTest:.075,alphaToCoverage:true,side:DoubleSide,fog:true,toneMapped:false});
  m.onBeforeCompile=s=>{s.vertexShader=s.vertexShader.replace('#include <project_vertex>',`
   vec3 centre=instanceMatrix[3].xyz;
   float treeScale=length(instanceMatrix[1].xyz);
   vec3 treeRight=normalize(vec3(cameraPosition.z-centre.z,0.,centre.x-cameraPosition.x));
   vec3 treePosition=centre+treeRight*position.x*treeScale+vec3(0.,position.y*treeScale,0.);
   vec4 mvPosition=modelViewMatrix*vec4(treePosition,1.);gl_Position=projectionMatrix*mvPosition;
  `);};return m;
 },[impostor]);
 useEffect(()=>{
  const o=new Object3D();
  placements.near.forEach((t,i)=>{o.position.set(t[0],t[1],t[2]);o.rotation.set(0,t[4],0);o.scale.setScalar(t[3]/14.4);o.updateMatrix();for(const mesh of near.current)mesh?.setMatrixAt(i,o.matrix);});
  for(const mesh of near.current)if(mesh){mesh.instanceMatrix.needsUpdate=true;mesh.computeBoundingSphere();}
  placements.far.forEach((t,i)=>{o.position.set(t[0],t[1],t[2]);o.rotation.set(0,0,0);o.scale.setScalar(t[3]/14.4);o.updateMatrix();far.current?.setMatrixAt(i,o.matrix);far.current?.setColorAt(i,new Color().setRGB(.73+(i%7)*.025,.82+(i%5)*.028,.75+(i%3)*.035));});
  if(far.current){far.current.instanceMatrix.needsUpdate=true;if(far.current.instanceColor)far.current.instanceColor.needsUpdate=true;far.current.computeBoundingSphere();}
 },[placements,impostor]);
 useEffect(()=>()=>{plane.dispose();billboard.dispose();},[plane,billboard]);
 return <group>{source.map((p,i)=><instancedMesh key={i} ref={m=>{near.current[i]=m;}} args={[p.geometry,p.material,placements.near.length]} castShadow receiveShadow/>)}{impostor&&<instancedMesh ref={far} args={[plane,billboard,placements.far.length]} frustumCulled={false}/>}</group>;
}
function firPlacements(hole:Hole,anchor:Vec3,high:boolean){const all=hole.data.trees.filter((_,i)=>i%3!==0);const near=all.filter(t=>Math.hypot(t[0]-anchor[0],t[2]-anchor[2])<72).sort((a,b)=>Math.hypot(a[0]-anchor[0],a[2]-anchor[2])-Math.hypot(b[0]-anchor[0],b[2]-anchor[2])).slice(0,high?4:2);return {near,far:all.filter(t=>!near.includes(t))};}
function NearFirs({hole,anchor,high}:{hole:Hole;anchor:Vec3;high:boolean}){
 const {scene}=useGLTF(ASSETS+(high?'fir-near.glb':'fir-near-mobile.glb')),meshes=useRef<(InstancedMesh|null)[]>([]);
 const parts=useMemo(()=>{scene.updateMatrixWorld(true);const box=new Box3().setFromObject(scene),scale=14.4/(box.max.y-box.min.y),center=box.getCenter(new Vector3()),parts:Mesh[]=[];scene.traverse(o=>{if(o instanceof Mesh){const m=o.clone();m.geometry=o.geometry.clone().applyMatrix4(o.matrixWorld).scale(scale,scale,scale).translate(-center.x*scale,-box.min.y*scale,-center.z*scale);m.material=(o.material as MeshStandardMaterial).clone();(m.material as MeshStandardMaterial).envMapIntensity=.45;parts.push(m);}});return parts;},[scene]);
 const trees=useMemo(()=>firPlacements(hole,anchor,high).near,[hole,anchor,high]);
 useEffect(()=>{const o=new Object3D();trees.forEach((t,i)=>{o.position.set(t[0],t[1],t[2]);o.rotation.set(0,t[4],0);o.scale.setScalar(t[3]/14.4);o.updateMatrix();meshes.current.forEach(m=>m?.setMatrixAt(i,o.matrix));});meshes.current.forEach(m=>{if(m){m.instanceMatrix.needsUpdate=true;m.computeBoundingSphere();}});},[trees]);
 useEffect(()=>()=>parts.forEach(m=>{m.geometry.dispose();(m.material as MeshStandardMaterial).dispose();}),[parts]);
 return <group>{parts.map((m,i)=><instancedMesh key={i} ref={r=>{meshes.current[i]=r;}} args={[m.geometry,m.material,trees.length]} castShadow receiveShadow/>)}</group>;
}
function FirForest({hole,anchor,high}:{hole:Hole;anchor:Vec3;high:boolean}){
 const atlas=useTexture(ASSETS+'fir-atlas.webp',t=>{t.colorSpace=SRGBColorSpace;t.anisotropy=8;});
 const mesh=useRef<InstancedMesh>(null);
  const trees=useMemo(()=>firPlacements(hole,anchor,high).far,[hole,anchor,high]);
 const geometry=useMemo(()=>new PlaneGeometry(8,15.3).translate(0,7.55,0),[]);
 const material=useMemo(()=>{
  const m=new MeshBasicMaterial({map:atlas,alphaTest:.13,alphaToCoverage:true,side:DoubleSide,fog:true,toneMapped:false});
  m.onBeforeCompile=s=>{
   s.vertexShader='varying vec2 treeTile;\n'+s.vertexShader;
   s.fragmentShader='varying vec2 treeTile;\n'+s.fragmentShader;
   s.vertexShader=s.vertexShader.replace('#include <project_vertex>',`
    vec3 centre=instanceMatrix[3].xyz;
    float treeScale=length(instanceMatrix[1].xyz);
    vec3 treeRight=normalize(vec3(cameraPosition.z-centre.z,0.,centre.x-cameraPosition.x));
    float angle=atan(cameraPosition.x-centre.x,cameraPosition.z-centre.z)+atan(instanceMatrix[0].z,instanceMatrix[0].x);
    float tile=mod(floor(angle/1.5707963+.5)+4.,4.);
    treeTile=vec2(mod(tile,2.),1.-floor(tile/2.));
    vec3 treePosition=centre+treeRight*position.x*length(instanceMatrix[0].xyz)+vec3(0.,position.y*treeScale,0.);
    vec4 mvPosition=modelViewMatrix*vec4(treePosition,1.);gl_Position=projectionMatrix*mvPosition;
   `);
   s.fragmentShader=s.fragmentShader.replace('#include <map_fragment>','diffuseColor *= texture2D(map,(vMapUv+treeTile)*.5);');
  };return m;
 },[atlas]);
 useEffect(()=>{
  if(!mesh.current)return;const o=new Object3D(),rand=random(421);
  trees.forEach((t,i)=>{o.position.set(t[0],t[1],t[2]);o.rotation.set(0,t[4],0);o.scale.set(t[3]/14.4*(.9+rand()*.35),t[3]/14.4,t[3]/14.4);o.updateMatrix();mesh.current!.setMatrixAt(i,o.matrix);const tint=.85+rand()*.15;mesh.current!.setColorAt(i,new Color().setRGB(tint,tint,tint*.96));});
  mesh.current.instanceMatrix.needsUpdate=true;if(mesh.current.instanceColor)mesh.current.instanceColor.needsUpdate=true;mesh.current.computeBoundingSphere();
 },[trees]);
 useEffect(()=>()=>{geometry.dispose();material.dispose();},[geometry,material]);
 return <instancedMesh ref={mesh} args={[geometry,material,trees.length]} frustumCulled={false}/>;
}
export const Woodland=memo(function Woodland(props:{hole:Hole;anchor:Vec3;high:boolean}){return <><FirForest {...props}/><Suspense fallback={null}><NearFirs {...props}/></Suspense><Suspense fallback={null}><TreeLayer {...props} birch/></Suspense></>;});

export const Turf=memo(function Turf({hole,anchor,high}:{hole:Hole;anchor:Vec3;high:boolean}){
 const mesh=useRef<InstancedMesh>(null),clock=useRef({value:0});
 const geometry=useMemo(()=>{const g=new BufferGeometry();g.setAttribute('position',new Float32BufferAttribute([-.004,0,0,.004,0,0,-.002,.55,.008,.002,.55,.008,.004,1,.014],3));g.setAttribute('uv',new Float32BufferAttribute([0,0,1,0,0,.5,1,.5,.5,1],2));g.setIndex([0,1,2,1,3,2,2,3,4]);g.computeVertexNormals();return g;},[]);
  const count=high?42000:12000;
 const material=useMemo(()=>{
  const m=new MeshStandardMaterial({color:'#ffffff',emissive:'#374a1c',emissiveIntensity:.25,roughness:1,side:DoubleSide});
  m.onBeforeCompile=s=>{s.uniforms.turfTime=clock.current;s.vertexShader='uniform float turfTime; varying float bladeHeight;\n'+s.vertexShader;s.fragmentShader='varying float bladeHeight;\n'+s.fragmentShader;s.vertexShader=s.vertexShader.replace('#include <begin_vertex>',`#include <begin_vertex>\nbladeHeight=position.y; transformed.x+=sin(turfTime*1.2+instanceMatrix[3].x*.27+instanceMatrix[3].z*.21)*position.y*position.y*.06;`);s.fragmentShader=s.fragmentShader.replace('#include <color_fragment>','#include <color_fragment>\ndiffuseColor.rgb*=.48+bladeHeight*.52;');};return m;
 },[]);
 useEffect(()=>{
  if(!mesh.current)return;const rand=random(1704),o=new Object3D(),nearRoad=hole.data.road.filter(p=>Math.hypot(p[0]-anchor[0],p[2]-anchor[2])<18);let used=0;
  for(let i=0;i<count*2&&used<count;i++){
   const a=rand()*Math.PI*2,r=Math.sqrt(rand())*14,x=anchor[0]+Math.cos(a)*r,z=anchor[2]+Math.sin(a)*r,lie=surfaceAt(hole.world,x,z);
   if(lie!=='rough'||nearRoad.some(p=>Math.hypot(p[0]-x,p[2]-z)<2.2))continue;
   const length=(.045+rand()*.045)*Math.min(1,(14-r)/4);
   o.position.set(x,heightAt(hole.world,x,z)-.004,z);o.rotation.set(0,rand()*Math.PI*2,0);o.scale.set(.45+rand()*.6,length,1);o.updateMatrix();mesh.current.setMatrixAt(used,o.matrix);mesh.current.setColorAt(used,new Color().setRGB(.16+rand()*.07,.23+rand()*.07,.048+rand()*.025));used++;
  }mesh.current.count=used;mesh.current.instanceMatrix.needsUpdate=true;if(mesh.current.instanceColor)mesh.current.instanceColor.needsUpdate=true;mesh.current.computeBoundingSphere();
 },[hole,anchor,count]);
 useFrame((_,dt)=>{clock.current.value+=dt;});
 useEffect(()=>()=>{geometry.dispose();material.dispose();},[geometry,material]);
 return <instancedMesh ref={mesh} args={[geometry,material,count]} receiveShadow/>;
});

export function TeeDetails({hole}:{hole:Hole}){
 const p=hole.data.tee;
 return <group>{[-2.2,2.2].map(x=><group key={x} position={[p[0]+x,p[1]+.035,p[2]-.2]}><mesh rotation={[0,.15,0]} castShadow><boxGeometry args={[.30,.105,.16]}/><meshStandardMaterial color="#e6dcc8" roughness={.78}/></mesh><mesh position={[0,.055,0]} rotation={[-Math.PI/2,0,0]}><circleGeometry args={[.045,24]}/><meshStandardMaterial color="#b49557" metalness={.45} roughness={.35}/></mesh></group>)}<group position={[p[0]-6.6,heightAt(hole.world,p[0]-6.6,p[2]+2),p[2]+2]} rotation={[0,-.1,0]}><mesh position={[0,.38,0]} castShadow><boxGeometry args={[.12,.8,.12]}/><meshStandardMaterial color="#5f5140" roughness={.95}/></mesh><mesh position={[0,.85,0]} castShadow><boxGeometry args={[.70,.40,.06]}/><meshStandardMaterial color="#183931" roughness={.68}/></mesh><mesh position={[0,.86,.034]}><planeGeometry args={[.59,.006]}/><meshStandardMaterial color="#cfbd90"/></mesh></group></group>;
}
