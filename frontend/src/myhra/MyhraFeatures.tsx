import {useEffect,useMemo} from 'react';
import {useFrame} from '@react-three/fiber';
import {useGLTF} from '@react-three/drei';
import {Box3,BufferGeometry,DataTexture,Float32BufferAttribute,Mesh,MeshPhysicalMaterial,MeshStandardMaterial,RepeatWrapping,RGBAFormat,Shape,ShapeGeometry,Vector2,Vector3,type Group} from 'three';
import {ASSETS,type Hole} from './data';
import {heightAt} from '../course/engine';

export function Pond({hole,high}:{hole:Hole;high:boolean}){
 const normal=useMemo(()=>{const n=128,a=new Uint8Array(n*n*4);for(let y=0;y<n;y++)for(let x=0;x<n;x++){const u=x/n*Math.PI*2,v=y/n*Math.PI*2,i=(y*n+x)*4;a[i]=128+Math.sin(u*3+Math.sin(v*2))*27+Math.sin(u*7+v*5)*12;a[i+1]=128+Math.cos(v*3+Math.sin(u*3))*25;a[i+2]=252;a[i+3]=255;}const t=new DataTexture(a,n,n,RGBAFormat);t.wrapS=t.wrapT=RepeatWrapping;t.repeat.set(16,16);t.needsUpdate=true;return t;},[]);
  const geometry=useMemo(()=>{const shape=new Shape(hole.data.pond.points.map(([x,z])=>new Vector2(x,-z))),g=new ShapeGeometry(shape),uv=g.getAttribute('uv');for(let i=0;i<uv.count;i++)uv.setXY(i,(uv.getX(i)-394)/60,(uv.getY(i)+500)/90);return g;},[hole]);
  const bank=useMemo(()=>{const points=hole.data.pond.points,cx=points.reduce((s,p)=>s+p[0],0)/points.length,cz=points.reduce((s,p)=>s+p[1],0)/points.length,positions:number[]=[],colors:number[]=[],indices:number[]=[];
   for(let i=0;i<points.length;i++){const a=points[i],b=points[(i+1)%points.length],steps=Math.max(2,Math.ceil(Math.hypot(b[0]-a[0],b[1]-a[1])*2));for(let j=0;j<steps;j++){const f=j/steps,x=a[0]+(b[0]-a[0])*f,z=a[1]+(b[1]-a[1])*f,dx=x-cx,dz=z-cz,len=Math.hypot(dx,dz),ox=x+dx/len*.7,oz=z+dz/len*.7;positions.push(x,hole.data.pond.height+.012,z,ox,Math.max(hole.data.pond.height+.035,heightAt(hole.world,ox,oz)+.009),oz);colors.push(.17,.20,.105,.28,.31,.13);}}
   const count=positions.length/6;for(let i=0;i<count;i++){const a=i*2,b=((i+1)%count)*2;indices.push(a,b,a+1,a+1,b,b+1);}const g=new BufferGeometry();g.setAttribute('position',new Float32BufferAttribute(positions,3));g.setAttribute('color',new Float32BufferAttribute(colors,3));g.setIndex(indices);g.computeVertexNormals();return g;
  },[hole]);
  useEffect(()=>()=>{geometry.dispose();normal.dispose();bank.dispose();},[geometry,normal,bank]);
 useFrame((_,dt)=>{normal.offset.set(normal.offset.x+dt*.003,normal.offset.y+dt*.002);});
  return <group><mesh geometry={geometry} rotation={[-Math.PI/2,0,0]} position={[0,hole.data.pond.height,0]}>
   <meshPhysicalMaterial color="#225a59" metalness={0} roughness={high?.17:.24} normalMap={normal} normalScale={new Vector2(.22,.22)} envMapIntensity={.7} clearcoat={.65} clearcoatRoughness={.15} ior={1.333}/>
  </mesh><mesh geometry={bank} receiveShadow><meshStandardMaterial vertexColors roughness={.9} side={2}/></mesh></group>;
}
export function GolfCart({actor,children,high=true}:{actor:React.RefObject<Group|null>;children?:React.ReactNode;high?:boolean}){
 const source=useGLTF(ASSETS+(high?'cart.glb':'cart-mobile.glb'));
 const model=useMemo(()=>{
  const scene=source.scene.clone(true);scene.updateMatrixWorld(true);const box=new Box3().setFromObject(scene),size=box.getSize(new Vector3()),center=box.getCenter(new Vector3()),scale=2.6/Math.max(size.x,size.y,size.z);
  scene.position.set(-center.x*scale,-box.min.y*scale,-center.z*scale);scene.scale.setScalar(scale);
  scene.traverse(o=>{if(o instanceof Mesh){o.castShadow=true;o.receiveShadow=true;const m=new MeshPhysicalMaterial();MeshStandardMaterial.prototype.copy.call(m,o.material as MeshStandardMaterial);m.clearcoat=.35;m.clearcoatRoughness=.22;m.envMapIntensity=1.2;if(m.name==='Clear windscreen'){m.envMapIntensity=.15;m.clearcoat=.08;m.depthWrite=false;o.castShadow=false;o.receiveShadow=false;}o.material=m;}});
  return scene;
 },[source.scene]);
 return <group ref={actor}><primitive object={model}/>{children}</group>;
}
