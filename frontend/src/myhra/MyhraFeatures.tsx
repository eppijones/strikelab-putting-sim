import {useEffect,useMemo} from 'react';
import {useFrame} from '@react-three/fiber';
import {useGLTF} from '@react-three/drei';
import {Box3,DataTexture,Mesh,MeshPhysicalMaterial,MeshStandardMaterial,RepeatWrapping,RGBAFormat,Shape,ShapeGeometry,Vector2,Vector3,type Group} from 'three';
import {ASSETS,type Hole} from './data';

export function Pond({hole,high}:{hole:Hole;high:boolean}){
 const normal=useMemo(()=>{const n=128,a=new Uint8Array(n*n*4);for(let y=0;y<n;y++)for(let x=0;x<n;x++){const u=x/n*Math.PI*2,v=y/n*Math.PI*2,i=(y*n+x)*4;a[i]=128+Math.sin(u*3+Math.sin(v*2))*27+Math.sin(u*7+v*5)*12;a[i+1]=128+Math.cos(v*3+Math.sin(u*3))*25;a[i+2]=252;a[i+3]=255;}const t=new DataTexture(a,n,n,RGBAFormat);t.wrapS=t.wrapT=RepeatWrapping;t.repeat.set(16,16);t.needsUpdate=true;return t;},[]);
 const geometry=useMemo(()=>{const shape=new Shape(hole.data.pond.points.map(([x,z])=>new Vector2(x,-z))),g=new ShapeGeometry(shape),uv=g.getAttribute('uv');for(let i=0;i<uv.count;i++)uv.setXY(i,(uv.getX(i)-394)/60,(uv.getY(i)+500)/90);return g;},[hole]);
 useEffect(()=>()=>{geometry.dispose();normal.dispose();},[geometry,normal]);
 useFrame((_,dt)=>{normal.offset.set(normal.offset.x+dt*.003,normal.offset.y+dt*.002);});
 return <mesh geometry={geometry} rotation={[-Math.PI/2,0,0]} position={[0,hole.data.pond.height,0]}>
  <meshPhysicalMaterial color="#30564c" metalness={.65} roughness={high?.13:.18} normalMap={normal} normalScale={new Vector2(.16,.16)} envMapIntensity={1.35} clearcoat={1}/>
 </mesh>;
}
export function GolfCart({actor,children}:{actor:React.RefObject<Group|null>;children?:React.ReactNode}){
 const source=useGLTF(ASSETS+'cart.glb');
 const model=useMemo(()=>{
  const scene=source.scene.clone(true);scene.updateMatrixWorld(true);const box=new Box3().setFromObject(scene),size=box.getSize(new Vector3()),center=box.getCenter(new Vector3()),scale=2.6/Math.max(size.x,size.y,size.z);
  scene.position.set(-center.x*scale,-box.min.y*scale,-center.z*scale);scene.scale.setScalar(scale);
  scene.traverse(o=>{if(o instanceof Mesh){o.castShadow=true;o.receiveShadow=true;const m=new MeshPhysicalMaterial();MeshStandardMaterial.prototype.copy.call(m,o.material as MeshStandardMaterial);m.clearcoat=.35;m.clearcoatRoughness=.22;m.envMapIntensity=1.2;if(m.name==='Clear windscreen'){m.envMapIntensity=.15;m.clearcoat=.08;m.depthWrite=false;o.castShadow=false;o.receiveShadow=false;}o.material=m;}});
  return scene;
 },[source.scene]);
 return <group ref={actor}><primitive object={model}/>{children}</group>;
}
