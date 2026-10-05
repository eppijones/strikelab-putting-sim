import {useEffect,useMemo,useRef} from 'react';
import {useFrame} from '@react-three/fiber';
import {BufferGeometry,Float32BufferAttribute,InstancedMesh,Object3D} from 'three';
import {heightAt,surfaceAt,type Vec3} from '../course/engine';
import {slopeAt} from './sceneMotion';
import type {Hole} from './data';

/** Contour lines and beads sample the exact surface used by putting physics. */
export function GreenReading({hole,ball}:{hole:Hole;ball:Vec3}){
 const beads=useRef<InstancedMesh>(null);
 const data=useMemo(()=>{
  const vertices:number[]=[],flows:{x:number;z:number;dx:number;dz:number;speed:number;offset:number}[]=[];
  const center:[number,number]=[(ball[0]+hole.data.pin[0])*.5,(ball[2]+hole.data.pin[2])*.5];
  const radius=Math.min(18,Math.max(8,Math.hypot(ball[0]-hole.data.pin[0],ball[2]-hole.data.pin[2])*.6));
  const x0=Math.floor(center[0]-radius),z0=Math.floor(center[1]-radius),n=Math.ceil(radius*2);
  for(let i=0;i<=n;i++)for(let j=0;j<n*2;j++)for(const axis of [0,1]){
   const x=x0+(axis?j*.5:i),z=z0+(axis?i:j*.5),bx=x+(axis?.5:0),bz=z+(axis?0:.5);
   if(surfaceAt(hole.world,x,z)!=='green'||surfaceAt(hole.world,bx,bz)!=='green')continue;
   vertices.push(x,heightAt(hole.world,x,z)+.012,z,bx,heightAt(hole.world,bx,bz)+.012,bz);
  }
  for(let x=x0+.5;x<x0+n;x+=1.5)for(let z=z0+.5;z<z0+n;z+=1.5){
   if(surfaceAt(hole.world,x,z)!=='green')continue;const [gx,gz]=slopeAt(hole.world,x,z),s=Math.hypot(gx,gz);
   if(s<.002)continue;
   flows.push({x,z,dx:-gx/s,dz:-gz/s,speed:Math.min(.8,.10+s*6),offset:Math.abs(Math.sin(x*17+z*31))});
  }
  const geometry=new BufferGeometry();geometry.setAttribute('position',new Float32BufferAttribute(vertices,3));return {geometry,flows};
 },[hole,ball]);
 useEffect(()=>()=>data.geometry.dispose(),[data]);
 const scratch=useMemo(()=>new Object3D(),[]);
 useFrame(({clock})=>{if(!beads.current)return;data.flows.forEach((f,i)=>{const travel=((clock.elapsedTime*f.speed+f.offset)%1-.5)*1.1,x=f.x+f.dx*travel,z=f.z+f.dz*travel;
  scratch.position.set(x,heightAt(hole.world,x,z)+.022,z);scratch.rotation.set(-Math.PI/2,0,0);scratch.scale.setScalar(surfaceAt(hole.world,x,z)==='green'?1:0);scratch.updateMatrix();beads.current!.setMatrixAt(i,scratch.matrix);
 });beads.current.instanceMatrix.needsUpdate=true;});
 return <group><lineSegments geometry={data.geometry} renderOrder={2}><lineBasicMaterial color="#cae3d0" transparent opacity={.53} depthWrite={false}/></lineSegments><instancedMesh ref={beads} args={[undefined,undefined,data.flows.length]} frustumCulled={false} renderOrder={3}><circleGeometry args={[.037,6]}/><meshBasicMaterial color="#ffe5a2" transparent opacity={.88} depthWrite={false}/></instancedMesh></group>;
}
