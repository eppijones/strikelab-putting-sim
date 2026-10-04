import {useMemo,useRef,type MutableRefObject} from 'react';
import {useFrame} from '@react-three/fiber';
import {useGLTF} from '@react-three/drei';
import {clone} from 'three/examples/jsm/utils/SkeletonUtils.js';
import {Bone,Group,Matrix4,Mesh,MeshStandardMaterial,Quaternion,SkinnedMesh,Vector3} from 'three';
import {ASSETS} from './data';
import {clubFrame,golfPose,IMPACT_DELAY} from './golfMotion';
import type {SwingController} from '../course/swing';

const S=1.86/.982574;
const v=(x=0,y=0,z=0)=>new Vector3(x,y,z);
const X=v(1),Y=v(0,1);
const smooth=(t:number)=>{t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);};
type Rest={bone:Bone;position:Vector3;quaternion:Quaternion;world:Quaternion};

/** Two-bone constraints keep both hands on one grip and shoes on the ground. */
function aimBone(bone:Bone,child:Bone,target:Vector3){
 const origin=bone.getWorldPosition(v()),from=child.getWorldPosition(v()).sub(origin).normalize(),to=target.clone().sub(origin).normalize();
 const q=new Quaternion().setFromUnitVectors(from,to).multiply(bone.getWorldQuaternion(new Quaternion()));
 bone.quaternion.copy(bone.parent!.getWorldQuaternion(new Quaternion()).invert().multiply(q));bone.updateWorldMatrix(false,true);
}
function limb(upper:Bone,lower:Bone,end:Bone,goal:Vector3,pole:Vector3){
 const a=upper.getWorldPosition(v()),b=lower.getWorldPosition(v()),c=end.getWorldPosition(v());
 const l1=a.distanceTo(b),l2=b.distanceTo(c),dir=goal.clone().sub(a),d=Math.min(l1+l2-.0001,Math.max(.0001,dir.length()));dir.normalize();
 const side=pole.clone().sub(a);side.addScaledVector(dir,-side.dot(dir)).normalize();
 const along=(l1*l1-l2*l2+d*d)/(2*d),height=Math.sqrt(Math.max(0,l1*l1-along*along));
 const elbow=a.clone().addScaledVector(dir,along).addScaledVector(side,height);
 aimBone(upper,lower,elbow);aimBone(lower,end,goal);
}

export function GolfClub({putter=false,iron=false}:{putter?:boolean;iron?:boolean}){
 // Metre-scale, separate graphite/chrome/rubber materials retain real highlights.
 const length=putter?.86:iron?.99:1.10;
 return <group>
  <mesh position={[0,-length*.5,0]} castShadow><cylinderGeometry args={[.005,.0035,length,12]}/><meshStandardMaterial color={putter||iron?'#c6cbd0':'#283139'} metalness={putter||iron?.92:.65} roughness={.21}/></mesh>
  <mesh position={[0,-.09,0]} castShadow><cylinderGeometry args={[.011,.0085,.25,16]}/><meshStandardMaterial color="#20272a" roughness={.78}/></mesh>
  {[0,.04,.08,.12,.16,.20].map(y=><mesh key={y} position={[0,-y+.02,0]}><torusGeometry args={[.010,.0007,4,12]}/><meshStandardMaterial color="#525a5b" roughness={.7}/></mesh>)}
  {putter?<group position={[0,-length,0]}><mesh castShadow><boxGeometry args={[.105,.024,.035]}/><meshStandardMaterial color="#bac3c8" metalness={.95} roughness={.22}/></mesh><mesh position={[0,.013,0]}><boxGeometry args={[.003,.001,.025]}/><meshStandardMaterial color="#fffdf1"/></mesh></group>:iron?<group position={[0,-length,0]}><mesh castShadow><boxGeometry args={[.083,.05,.012]}/><meshStandardMaterial color="#d2d6d9" metalness={.96} roughness={.2}/></mesh>{[0,1,2,3,4].map(n=><mesh key={n} position={[0,-.016+n*.007,.0065]}><boxGeometry args={[.067,.0007,.001]}/><meshStandardMaterial color="#59616a"/></mesh>)}</group>:<group position={[0,-length,0]}><mesh scale={[1,.59,.84]} castShadow><sphereGeometry args={[.065,24,16]}/><meshPhysicalMaterial color="#132123" metalness={.8} roughness={.22} clearcoat={1}/></mesh><mesh position={[0,-.002,.047]} rotation={[.18,0,0]}><boxGeometry args={[.093,.045,.006]}/><meshStandardMaterial color="#aeb6bd" metalness={.95} roughness={.24}/></mesh></group>}
 </group>;
}

export default function MyhraGolfer({swing,elapsed,club,seated=false,feet}:{swing:MutableRefObject<SwingController>;elapsed:MutableRefObject<number>;club:number;seated?:boolean;feet?:MutableRefObject<[number,number]>}){
 const phaseMemory=useRef({back:0,transition:0,playing:false});
 const source=useGLTF(ASSETS+'golfer.glb'),grip=useRef<Group>(null);
 const rig=useMemo(()=>{
  const scene=clone(source.scene);scene.scale.setScalar(S);
  scene.traverse(o=>{if(o instanceof SkinnedMesh)o.skeleton.pose();if(o instanceof Mesh){o.castShadow=true;o.receiveShadow=true;o.frustumCulled=false;o.material=(o.material as MeshStandardMaterial).clone();(o.material as MeshStandardMaterial).envMapIntensity=.6;}});
  scene.updateMatrixWorld(true);
  const bones:Record<string,Bone>={},rest:Rest[]=[];
  scene.traverse(o=>{if(o instanceof Bone){bones[o.name.replace('mixamorig','').replace(':','')]=o;rest.push({bone:o,position:o.position.clone(),quaternion:o.quaternion.clone(),world:o.getWorldQuaternion(new Quaternion())});}});
  return {scene,bones,rest};
 },[source.scene]);
 useFrame(()=>{
  const {scene,bones,rest}=rig,s=swing.current,t=elapsed.current,putt=club===13;
  for(const r of rest){r.bone.position.copy(r.position);r.bone.quaternion.copy(r.quaternion);}
  scene.updateWorldMatrix(true,true);
  const rootQ=scene.getWorldQuaternion(new Quaternion()),toWorld=(p:Vector3)=>scene.localToWorld(p.clone().multiplyScalar(1/S));
  const rotation=(name:string,turn:number,lean:number)=>{const bone=bones[name],r=rest.find(r=>r.bone===bone)!;
   const q=rootQ.clone().multiply(new Quaternion().setFromAxisAngle(Y,turn)).multiply(new Quaternion().setFromAxisAngle(X,lean)).multiply(r.world);
   bone.quaternion.copy(bone.parent!.getWorldQuaternion(new Quaternion()).invert().multiply(q));bone.updateWorldMatrix(false,true);
  };
  if(seated){
   bones.Hips.position.copy(toWorld(v(0,.78,-.08)).applyMatrix4(bones.Hips.parent!.matrixWorld.clone().invert()));
   rotation('Hips',0,0);rotation('Spine',0,.04);rotation('Spine2',0,.08);rotation('Head',0,0);
   for(const side of ['Left','Right']){const sign=side==='Left'?1:-1;
    limb(bones[side+'UpLeg'],bones[side+'Leg'],bones[side+'Foot'],toWorld(v(sign*.14,.18,.52)),toWorld(v(sign*.22,.67,.75)));
    limb(bones[side+'Arm'],bones[side+'ForeArm'],bones[side+'Hand'],toWorld(v(sign*.14,1.10,.49)),toWorld(v(sign*.37,1.02,.22)));
   }return;
  }
  let back=0,follow=0;
  const memory=phaseMemory.current;
  if(t>=0){if(!memory.playing)memory.transition=memory.back;memory.playing=true;back=memory.transition*(1-smooth(t/IMPACT_DELAY));follow=smooth((t-IMPACT_DELAY)/(putt?.65:.95));}
  else {memory.playing=false;if(s.phase==='backswing'||s.phase==='downswing'||s.phase==='power')back=Math.min(1,s.amount);else if(s.phase==='path'||s.phase==='tempo')back=Math.min(1,s.lockedPower);memory.back=back;}
  const pose=golfPose(club,back,follow,t>=IMPACT_DELAY),turn=pose.chestTurn,hipTurn=pose.hipTurn,lean=pose.lean;
  bones.Hips.position.copy(toWorld(v(pose.weight,.90+follow*.07,-.04)).applyMatrix4(bones.Hips.parent!.matrixWorld.clone().invert()));
  rotation('Hips',hipTurn,.12*(1-follow));rotation('Spine',turn*.6,lean);rotation('Spine2',turn,lean);rotation('Head',turn*Math.max(0,follow-.25),.31*(1-follow));
  for(const side of ['Left','Right']){
   const sign=side==='Left'?1:-1,heel=side==='Right'?pose.heel:0;
   limb(bones[side+'UpLeg'],bones[side+'Leg'],bones[side+'Foot'],toWorld(v(sign*.235,.115+heel+(feet?.current[side==='Left'?0:1]??0),.01)),toWorld(v(sign*.26,.5,.8)));
  }
  const hands=pose.hands,shaft=pose.shaft;
  const left=hands.clone().addScaledVector(shaft,-.035).add(v(.058,.018,-.018)),right=hands.clone().addScaledVector(shaft,.025).add(v(-.058,.030,-.025));
  for(const side of ['Left','Right']){
   const goal=side==='Left'?left:right;
   limb(bones[side+'Arm'],bones[side+'ForeArm'],bones[side+'Hand'],toWorld(goal),toWorld(v(side==='Left'?.28:-.52,1.02,side==='Left'?.36:.08)));
   const hand=bones[side+'Hand'],fingers=hands.clone().sub(goal).normalize(),across=fingers.clone().cross(shaft).normalize(),palm=across.clone().cross(fingers).normalize();
   if(side==='Right'){across.negate();palm.negate();}
   const q=rootQ.clone().multiply(new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(across,fingers,palm)));
   hand.quaternion.copy(hand.parent!.getWorldQuaternion(new Quaternion()).invert().multiply(q));
   for(const finger of ['Index','Middle','Ring','Pinky'])for(let joint=1;joint<=3;joint++){
    const b=bones[side+'Hand'+finger+joint];if(b)b.quaternion.multiply(new Quaternion().setFromAxisAngle(X,joint===1?-.25:joint===2?-.9:-.6));
   }
  }
  if(grip.current){grip.current.position.copy(hands);grip.current.quaternion.copy(clubFrame(shaft,pose.faceTurn));}
 });
 return <group><primitive object={rig.scene}/>{!seated&&<group ref={grip}><GolfClub putter={club===13} iron={club>=3&&club<13}/></group>}</group>;
}
