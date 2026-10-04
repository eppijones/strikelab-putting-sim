import {useMemo,useRef} from 'react';
import {useGLTF} from '@react-three/drei';
import {useFrame} from '@react-three/fiber';
import {Bone,Group,Mesh,MeshStandardMaterial,Quaternion,Vector3} from 'three';
import {clone} from 'three/examples/jsm/utils/SkeletonUtils.js';
import {COURSE_ROOT} from './terrain';
import type {SwingController} from './swing';

interface Pose { walking:number; time:number; swing:SwingController; follow:number; putting:boolean }
const up=new Vector3(0,1,0);
function alignBone(bone:Bone,child:Bone,target:Vector3){
 const p=bone.getWorldPosition(new Vector3()),end=child.getWorldPosition(new Vector3());
 const q=new Quaternion().setFromUnitVectors(end.sub(p).normalize(),target.clone().sub(p).normalize());
 const world=bone.getWorldQuaternion(new Quaternion()).premultiply(q);
 const parent=bone.parent!.getWorldQuaternion(new Quaternion()).invert();bone.quaternion.copy(parent.multiply(world));bone.updateWorldMatrix(false,true);
}
function armIK(scene:Group,names:Map<string,Bone>,side:string,target:Vector3,bend:Vector3){
 const upper=names.get('upperarm_'+side),lower=names.get('lowerarm_'+side),hand=names.get('hand_'+side);if(!upper||!lower||!hand)return;
 const s=upper.getWorldPosition(new Vector3()),e=lower.getWorldPosition(new Vector3()),h=hand.getWorldPosition(new Vector3());
 const l1=s.distanceTo(e),l2=e.distanceTo(h),dir=target.clone().sub(s),d=Math.min(dir.length(),l1+l2-.003);dir.normalize();
 const along=(l1*l1-l2*l2+d*d)/(2*Math.max(d,.01)),height=Math.sqrt(Math.max(0,l1*l1-along*along));
 const bendDir=scene.localToWorld(bend.clone()).sub(s);bendDir.addScaledVector(dir,-bendDir.dot(dir)).normalize();
 const elbow=s.clone().addScaledVector(dir,along).addScaledVector(bendDir,height);
 alignBone(upper,lower,elbow);alignBone(lower,hand,target);
}

export default function Golfer({actor,pose}:{actor:React.RefObject<Group|null>;pose:React.RefObject<Pose>}){
 const files=useGLTF(['golfer-body-clean','golfer-face','golfer-outfit-clean'].map(n=>COURSE_ROOT+'art/'+n+'.glb'));
 const parts=useMemo(()=>files.map((file,index)=>{
  const scene=clone(file.scene) as Group,names=new Map<string,Bone>(),rest=new Map<Bone,Quaternion>();
  let meshIndex=0;
  scene.traverse(o=>{if(o instanceof Mesh){o.castShadow=true;o.receiveShadow=true;if(index===0)o.material=new MeshStandardMaterial({color:'#b88563',roughness:.72});if(index===2)o.material=new MeshStandardMaterial({color:meshIndex++===0?'#235144':'#c6ba9f',roughness:.9});}if(o instanceof Bone){names.set(o.name,o);rest.set(o,o.quaternion.clone());}});
  return {scene,names,rest};
 }),[files]);
 const shaft=useRef<Mesh>(null),head=useRef<Group>(null),leftShoe=useRef<Mesh>(null),rightShoe=useRef<Mesh>(null);
 useFrame(()=>{
  const p=pose.current,phase=p.swing.phase;
  const swingAmount=phase==='backswing'||phase==='downswing'?p.swing.amount:0;
  const f=p.follow;
  const angle=p.putting?(-swingAmount*.3+f*.32):(-swingAmount*1.25+f*1.5);
  const grip=new Vector3(Math.sin(angle)*.33,.92+(1-Math.cos(angle))*.65,.43);
  if(p.walking){grip.set(-.1,.9,.1);}
  const clubEnd=grip.clone().add(new Vector3(Math.sin(angle)*.9,-Math.cos(angle)*.9,.24));
  for(const part of parts){
   for(const [bone,q] of part.rest)bone.quaternion.copy(q);
   part.scene.updateMatrixWorld(true);
   if(p.walking){
    for(const side of ['l','r']){
     const upper=part.names.get('thigh_'+side),lower=part.names.get('calf_'+side),foot=part.names.get('foot_'+side);
     const stride=Math.sin(p.time*7+(side==='l'?0:Math.PI))*Math.min(1,p.walking);
     if(upper&&lower&&foot){
      const local=new Vector3(side==='l'?.13:-.13,.5,.02+stride*.2);alignBone(upper,lower,part.scene.localToWorld(local));
      const footTarget=part.scene.localToWorld(new Vector3(side==='l'?.13:-.13,.085+Math.max(0,-stride)*.12,stride*.32));alignBone(lower,foot,footTarget);
     }
     const arm=part.names.get('upperarm_'+side),elbow=part.names.get('lowerarm_'+side),hand=part.names.get('hand_'+side);
     if(arm&&elbow&&hand){alignBone(arm,elbow,part.scene.localToWorld(new Vector3(side==='l'?.22:-.22,1.15,-stride*.13)));alignBone(elbow,hand,part.scene.localToWorld(new Vector3(side==='l'?.23:-.23,.9,-stride*.23)));}
    }
   }else{
    armIK(part.scene,part.names,'l',part.scene.localToWorld(grip.clone().add(new Vector3(.035,.025,0))),new Vector3(.35,1.15,.45));
    armIK(part.scene,part.names,'r',part.scene.localToWorld(grip.clone().add(new Vector3(-.035,-.025,0))),new Vector3(-.35,1.15,.45));
   }
  }
  if(actor.current)for(const [side,shoe] of [['l',leftShoe.current],['r',rightShoe.current]] as const){const foot=parts[0].names.get('foot_'+side);if(foot&&shoe)shoe.position.copy(actor.current.worldToLocal(foot.getWorldPosition(new Vector3()))).add(new Vector3(0,-.035,.065));}
  if(shaft.current){shaft.current.visible=!p.walking;shaft.current.position.copy(grip).add(clubEnd).multiplyScalar(.5);shaft.current.quaternion.setFromUnitVectors(up,grip.clone().sub(clubEnd).normalize());shaft.current.scale.y=grip.distanceTo(clubEnd);}
  if(head.current){head.current.visible=!p.walking;head.current.position.copy(clubEnd);head.current.rotation.z=-angle;}
 });
 return <group ref={actor}>
  {parts.map((p,i)=><primitive key={i} object={p.scene}/>)}
  {[leftShoe,rightShoe].map((ref,i)=><mesh key={i} ref={ref} scale={[.09,.052,.17]} castShadow><sphereGeometry args={[1,16,10]}/><meshStandardMaterial color="#e7e5da" roughness={.65}/></mesh>)}
  <mesh position={[0,1.735,.015]} scale={[1.04,.55,1]} castShadow><sphereGeometry args={[.106,20,10,0,Math.PI*2,0,Math.PI/2]}/><meshStandardMaterial color="#f2efe2" roughness={.75}/></mesh>
  <mesh position={[0,1.73,.115]} rotation={[-.04,0,0]} scale={[1,.1,1]} castShadow><sphereGeometry args={[.11,16,8]}/><meshStandardMaterial color="#f2efe2"/></mesh>
  <mesh ref={shaft} castShadow><cylinderGeometry args={[.006,.004,1,8]}/><meshStandardMaterial color="#bac4cd" metalness={.85} roughness={.24}/></mesh>
  <group ref={head}><mesh castShadow><boxGeometry args={[.11,.045,.055]}/><meshStandardMaterial color="#a7b0b3" metalness={.9} roughness={.2}/></mesh></group>
 </group>;
}
export type {Pose};
