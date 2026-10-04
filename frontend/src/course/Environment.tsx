import {blocksTeeCorridor} from './teeClearance';
import {memo,useEffect,useMemo,useRef} from 'react';
import {useFrame} from '@react-three/fiber';
import {useGLTF,useTexture} from '@react-three/drei';
import {BufferGeometry,Float32BufferAttribute,Color,DoubleSide,InstancedMesh,Mesh,MeshStandardMaterial,Object3D,RepeatWrapping,SRGBColorSpace,Vector2,Vector3} from 'three';
import {terrainGeometry,COURSE_ROOT} from './terrain';
import {heightAt,inPolygon,surfaceAt,type Course,type Vec3,type World} from './engine';
const ART=COURSE_ROOT+'art/';
function randomGenerator(seed:number){return()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};}

export const Terrain=memo(function Terrain({world}:{world:World}){
 const textures=useTexture(['fairway','green','rough','sand','fairway-normal'].map(n=>ART+n+'.webp'),loaded=>{
  loaded.forEach((t,i)=>{t.wrapS=t.wrapT=RepeatWrapping;t.anisotropy=8;if(i<4)t.colorSpace=SRGBColorSpace;else t.repeat.set(2400,2400);});
 });
 const mask=useTexture(ART+'surface-mask.png');
 const extent=(world.terrain.size-1)*world.terrain.spacing;
 const coarse=useMemo(()=>terrainGeometry(world.terrain,world.detail,extent),[world.terrain,world.detail,extent]);
 const detail=useMemo(()=>world.detail?terrainGeometry(world.detail,undefined,extent):undefined,[world.detail,extent]);
 const material=useMemo(()=>{
  const m=new MeshStandardMaterial({color:'#ffffff',roughness:.96,normalMap:textures[4],normalScale:new Vector2(.3,.3)});
  m.onBeforeCompile=shader=>{
   Object.assign(shader.uniforms,{grassMap:{value:textures[0]},greenMap:{value:textures[1]},roughMap:{value:textures[2]},sandMap:{value:textures[3]},surfaceMap:{value:mask},courseExtent:{value:extent}});
   shader.vertexShader='varying vec3 vGround;\n'+shader.vertexShader;
   shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\nvGround=position;');
   shader.fragmentShader='varying vec3 vGround; uniform sampler2D grassMap,greenMap,roughMap,sandMap,surfaceMap; uniform float courseExtent;\n'+shader.fragmentShader;
   shader.fragmentShader=shader.fragmentShader.replace('#include <map_fragment>',`
    vec3 surfaces=texture2D(surfaceMap,vec2(vGround.x/courseExtent,1.0-vGround.z/courseExtent)).rgb;
    float bunker=clamp(1.0-surfaces.r-surfaces.g-surfaces.b,0.0,1.0);
    vec3 grass=texture2D(grassMap,vGround.xz*1.2).rgb;
    vec3 green=texture2D(greenMap,vGround.xz*2.0).rgb;
    vec3 rough=texture2D(roughMap,vGround.xz*.85).rgb;
    vec3 sand=texture2D(sandMap,vGround.xz*.8).rgb;
    float mow=sin((vGround.x+vGround.z*.6)*.5)*.045;
    vec3 ground=rough*surfaces.r+(grass*(1.0+mow))*surfaces.g+(green*(1.06+mow*.7))*surfaces.b+sand*bunker;
    float variation=1.0+.025*sin(vGround.x*.14)*sin(vGround.z*.11);
    diffuseColor.rgb*=ground*variation;
   `);
  };m.customProgramCacheKey=()=> 'grenland-ground-v2';return m;
 },[textures,mask,extent]);
 useEffect(()=>()=>{coarse.dispose();detail?.dispose();material.dispose();},[coarse,detail,material]);
 return <group><mesh geometry={coarse} material={material} receiveShadow/>{detail&&<mesh geometry={detail} material={material} receiveShadow/>}</group>;
});

export const Trees=memo(function Trees({course,world,anchor,quality}:{course:Course;world:World;anchor:Vec3;quality:'balanced'|'high'}){
 const {scene}=useGLTF(ART+'maple-clean.glb');
 const textures=useTexture([ART+'maple-bark.webp',ART+'maple-leaf.webp'],ts=>ts.forEach(t=>{t.colorSpace=SRGBColorSpace;t.anisotropy=4;t.flipY=false;}));
 const trunk=useRef<InstancedMesh>(null),leaves=useRef<InstancedMesh>(null);
 const geometries=useMemo(()=>{
  const meshes:Mesh[]=[];scene.traverse(o=>{if(o instanceof Mesh)meshes.push(o);});
  if(meshes.length<2)return [];
  // PVE's reduced mesh stores one atlas sample per triangle. Rebuild leaf cards
  // with explicit UVs so browser foliage retains the source tree's crown.
  const source=meshes[1].geometry,p=source.getAttribute('position'),idx=source.index;
  const positions:number[]=[],uvs:number[]=[],indices:number[]=[];
  const a=new Vector3(),b=new Vector3(),c=new Vector3();
  for(let i=0;i<(idx?.count??p.count);i+=3){
   a.fromBufferAttribute(p,idx?idx.getX(i):i);b.fromBufferAttribute(p,idx?idx.getX(i+1):i+1);c.fromBufferAttribute(p,idx?idx.getX(i+2):i+2);
   const center=a.clone().add(b).add(c).multiplyScalar(1/3),u=b.clone().sub(a),normal=u.clone().cross(c.clone().sub(a)).normalize();
   const size=Math.max(.35,Math.min(.8,u.length()*1.5));u.normalize().multiplyScalar(size);const v=normal.clone().cross(u).normalize().multiplyScalar(size);
   const base=positions.length/3;for(const [x,y] of [[-1,-1],[1,-1],[-1,1],[1,1]])positions.push(...center.clone().addScaledVector(u,x*.5).addScaledVector(v,y*.5).toArray());
   uvs.push(0,0,1,0,0,1,1,1);indices.push(base,base+1,base+2,base+1,base+3,base+2);
  }
  const leaves=new BufferGeometry();leaves.setAttribute('position',new Float32BufferAttribute(positions,3));leaves.setAttribute('uv',new Float32BufferAttribute(uvs,2));leaves.setIndex(indices);leaves.computeVertexNormals();
  return [meshes[0].geometry,leaves];
 },[scene]);
 const placements=useMemo(()=>{
  const random=randomGenerator(732),trees:{x:number;z:number;y:number;s:number;angle:number;d:number}[]=[];
  for(const region of course.regions.filter(r=>r.kind==='trees')){
   const xs=region.points.map(p=>p[0]),zs=region.points.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),z0=Math.min(...zs),z1=Math.max(...zs);
   const count=Math.min(250,Math.ceil((x1-x0)*(z1-z0)/65));
   for(let i=0;i<count;i++){
    const x=x0+random()*(x1-x0),z=z0+random()*(z1-z0),d=Math.hypot(x-anchor[0],z-anchor[2]);
    const s=.52+random()*.46,angle=random()*6.28;
    if(d>850||!inPolygon(x,z,region.points)||course.practice.some(h=>blocksTeeCorridor(x,z,h)||Math.hypot(x-h.tee[0],z-h.tee[2])<13||Math.hypot(x-h.pin[0],z-h.pin[2])<14))continue;
    trees.push({x,z,y:heightAt(world,x,z),s,angle,d});
   }
  }
  return trees.sort((a,b)=>a.d-b.d).slice(0,quality==='high'?380:190);
 },[course,world,anchor,quality]);
 useEffect(()=>{const obj=new Object3D();placements.forEach((t,i)=>{
  obj.position.set(t.x,t.y,t.z);obj.scale.setScalar(t.s);obj.rotation.y=t.angle;obj.updateMatrix();trunk.current?.setMatrixAt(i,obj.matrix);leaves.current?.setMatrixAt(i,obj.matrix);
  leaves.current?.setColorAt(i,new Color().setHSL(.23+(i%5)*.012,.16,.94));
 });for(const mesh of [trunk.current,leaves.current])if(mesh){mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;mesh.computeBoundingSphere();}},[placements]);
 if(geometries.length<2)return null;
 return <group><instancedMesh ref={trunk} args={[geometries[0],undefined,placements.length]} castShadow receiveShadow><meshStandardMaterial map={textures[0]} roughness={1}/></instancedMesh><instancedMesh ref={leaves} args={[geometries[1],undefined,placements.length]} castShadow receiveShadow><meshStandardMaterial map={textures[1]} alphaTest={.45} side={DoubleSide} roughness={.85}/></instancedMesh></group>;
});

export const Grass=memo(function Grass({world,anchor,quality}:{world:World;anchor:Vec3;quality:'balanced'|'high'}){
 const mesh=useRef<InstancedMesh>(null),clock=useRef({value:0});
 const geometry=useMemo(()=>{
  const positions:number[]=[],uv:number[]=[],indices:number[]=[];
  for(let i=0;i<3;i++){const a=i*Math.PI/3,dx=Math.cos(a)*.005,dz=Math.sin(a)*.005,b=positions.length/3;
   positions.push(-dx,0,-dz,dx,0,dz,-dx*.6,.5,-dz*.6,dx*.6,.5,dz*.6,.025,1,.015);uv.push(0,0,1,0,0,.5,1,.5,.5,1);indices.push(b,b+1,b+2,b+1,b+3,b+2,b+2,b+3,b+4);
  }const g=new BufferGeometry();g.setAttribute('position',new Float32BufferAttribute(positions,3));g.setAttribute('uv',new Float32BufferAttribute(uv,2));g.setIndex(indices);g.computeVertexNormals();return g;
 },[]);
 const count=quality==='high'?16000:7000;
 const material=useMemo(()=>{
  const m=new MeshStandardMaterial({color:'#8ba765',side:DoubleSide,roughness:1});
  m.onBeforeCompile=s=>{s.uniforms.windTime=clock.current;s.vertexShader='uniform float windTime;\n'+s.vertexShader;s.vertexShader=s.vertexShader.replace('#include <begin_vertex>',`#include <begin_vertex>\ntransformed.x += sin(windTime*1.3+instanceMatrix[3].x*.45+instanceMatrix[3].z*.31)*position.y*position.y*.12;`);};return m;
 },[]);
 useEffect(()=>{
  if(!mesh.current)return;const random=randomGenerator(92),o=new Object3D();let used=0;
  for(let i=0;i<count*3&&used<count;i++){
   const angle=random()*6.28,r=2+Math.sqrt(random())*32,x=anchor[0]+Math.cos(angle)*r,z=anchor[2]+Math.sin(angle)*r,lie=surfaceAt(world,x,z);
   if(lie==='bunker'||lie==='water'||lie==='out'||lie==='green')continue;
   const h=lie==='fairway'?.025+random()*.025:.10+random()*.18;
   o.position.set(x,heightAt(world,x,z)-.004,z);o.rotation.y=random()*6.28;o.scale.set(.7+random(),h,.7+random());o.updateMatrix();mesh.current.setMatrixAt(used,o.matrix);mesh.current.setColorAt(used,new Color().setHSL(.23+random()*.05,.45,.3+random()*.18));used++;
  }mesh.current.count=used;mesh.current.instanceMatrix.needsUpdate=true;if(mesh.current.instanceColor)mesh.current.instanceColor.needsUpdate=true;mesh.current.computeBoundingSphere();
 },[world,anchor,count]);
 useFrame((_,dt)=>{clock.current.value+=dt;});
 useEffect(()=>()=>{geometry.dispose();material.dispose();},[geometry,material]);
 return <instancedMesh ref={mesh} args={[geometry,material,count]} receiveShadow/>;
});
