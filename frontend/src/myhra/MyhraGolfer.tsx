import {useEffect,useLayoutEffect,useMemo,useRef,type MutableRefObject} from 'react';
import {useFrame} from '@react-three/fiber';
import {useGLTF} from '@react-three/drei';
import {clone} from 'three/examples/jsm/utils/SkeletonUtils.js';
import {AnimationMixer,Bone,Group,LoopOnce,Matrix4,Mesh,MeshStandardMaterial,Quaternion,SkinnedMesh,Vector3} from 'three';
import {ASSETS} from './data';
import {clubFrame,golfPose,IMPACT_DELAY} from './golfMotion';
import type {SwingController} from '../course/swing';

const S=1.86/.982574;
const v=(x=0,y=0,z=0)=>new Vector3(x,y,z);
const X=v(1),Y=v(0,1);
const smooth=(t:number)=>{t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);};
type Rest={bone:Bone;position:Vector3;quaternion:Quaternion;world:Quaternion;scale:Vector3};

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

export default function MyhraGolfer({swing,elapsed,club,seated=false,feet,replay=false,shotType='full',high=true}:{swing:MutableRefObject<SwingController>;elapsed:MutableRefObject<number>;club:number;seated?:boolean;feet?:MutableRefObject<[number,number]>;replay?:boolean;shotType?:'full'|'chip'|'pitch'|'putt';high?:boolean}){
 const phaseMemory=useRef({back:0,transition:0,playing:false});
 const source=useGLTF(ASSETS+(high?'golfer-motions.glb':'golfer-motions-mobile.glb')),grip=useRef<Group>(null);
 const rig=useMemo(()=>{
  const scene=clone(source.scene);scene.scale.setScalar(S);
  scene.traverse(o=>{if(o instanceof SkinnedMesh)o.skeleton.pose();if(o instanceof Mesh){o.castShadow=true;o.receiveShadow=true;o.frustumCulled=false;o.material=(o.material as MeshStandardMaterial).clone();(o.material as MeshStandardMaterial).envMapIntensity=.6;}});
  scene.updateMatrixWorld(true);
  const bones:Record<string,Bone>={},rest:Rest[]=[];
  scene.traverse(o=>{if(o instanceof Bone){bones[o.name.replace('mixamorig','').replace(':','')]=o;rest.push({bone:o,position:o.position.clone(),quaternion:o.quaternion.clone(),world:o.getWorldQuaternion(new Quaternion()),scale:o.scale.clone()});}});
  const grips:Record<string,Quaternion>={},calibration:Record<string,Quaternion>={},gripCenters:Record<string,Vector3>={};
  // These local finger rotations are authored and exported by Blender. They
  // retain each generated bone's actual axes, including mirrored right fingers.
  for(const track of source.animations.find(a=>a.name==='GolfGrip')?.tracks??[]){if(!track.name.endsWith('.quaternion'))continue;const name=track.name.split('.')[0].replace('mixamorig','').replace(':','');if(name.includes('Hand')&&name!=='LeftHand'&&name!=='RightHand')grips[name]=new Quaternion().fromArray(track.values,0);}
  for(const side of ['Left','Right']){const hand=bones[side+'Hand'];
   const local=(name:string)=>hand.worldToLocal(bones[side+'Hand'+name].getWorldPosition(v()));
   const forward=local('Middle1').normalize(),across=local('Index1').sub(local('Pinky1'));across.addScaledVector(forward,-across.dot(forward)).normalize();
   calibration[side]=new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(across,forward,across.clone().cross(forward).normalize())).invert();
  }
  for(const [name,q] of Object.entries(grips))bones[name]?.quaternion.copy(q);
  scene.updateMatrixWorld(true);
  for(const side of ['Left','Right']){const hand=bones[side+'Hand'],center=v();for(const name of ['Middle1','Middle3','Middle4','Ring1','Ring3','Ring4'])center.add(hand.worldToLocal(bones[side+'Hand'+name].getWorldPosition(v())));gripCenters[side]=center.multiplyScalar(1/6);}
  const mixer=new AnimationMixer(scene),actions=Object.fromEntries(source.animations.filter(a=>a.name.startsWith('Golf')).map(clip=>{const base=clip.clone();base.name+='Address';const action=mixer.clipAction(clip),address=mixer.clipAction(base);for(const a of [action,address]){a.setLoop(LoopOnce,1);a.clampWhenFinished=true;a.paused=true;a.play();}return [clip.name,{action,address,duration:clip.duration}];}));
  return {scene,bones,rest,grips,calibration,gripCenters,mixer,actions,anchor:scene.getObjectByName('GolfClubAnchor')};
 },[source.scene,source.animations]);
 const mutableRig=useRef(rig);useLayoutEffect(()=>{mutableRig.current=rig;},[rig]);
 useEffect(()=>()=>{rig.mixer.stopAllAction();rig.scene.traverse(o=>{if(o instanceof Mesh){const materials=Array.isArray(o.material)?o.material:[o.material];materials.forEach(m=>m.dispose());}});},[rig]);
 useFrame(()=>{
  const rig=mutableRig.current,{scene,bones,rest}=rig,s=swing.current,t=elapsed.current,putt=club===13;
  // AnimationMixer owns baked bones. Resetting them outside the mixer would
  // erase a paused/scrubbed pose when its cached values have not changed.
  if(seated||!rig.anchor)for(const r of rest){r.bone.position.copy(r.position);r.bone.quaternion.copy(r.quaternion);r.bone.scale.copy(r.scale);}
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
  if(t>=0){if(!memory.playing)memory.transition=s.phase==='finish'?s.amount:Math.max(memory.back,s.lockedPower||0);memory.playing=true;back=(replay?Math.max(.35,s.lockedPower||1):memory.transition)*(1-smooth(t/IMPACT_DELAY));follow=smooth((t-IMPACT_DELAY)/(putt?.65:.95));}
  else {memory.playing=false;if(s.phase==='backswing'||s.phase==='power')back=Math.min(1,s.amount);else if(s.phase==='downswing')back=Math.min(1,s.peak);else if(s.phase==='path'||s.phase==='tempo')back=Math.min(1,s.lockedPower);memory.back=back;}
  const pose=golfPose(club,back,follow,t>=IMPACT_DELAY),turn=pose.chestTurn,hipTurn=pose.hipTurn,lean=pose.lean;
  const family=putt?'GolfPutt':shotType==='chip'||shotType==='pitch'?'GolfChip':club>=3?'GolfIron':'GolfDriver',authored=rig.actions[family];
  if(authored&&rig.anchor){
   for(const [name,tracks] of Object.entries(rig.actions)){tracks.action.enabled=name===family;tracks.address.enabled=name===family;}
   const strength=replay?1:Math.max(.01,memory.transition),impactTime=1+IMPACT_DELAY;
   // A partial backswing blends address/top, but every downswing reaches the
   // same square contact pose. Scaling the impact pose left weak putts short
   // of the ball. Retiming the follow-through keeps a shorter stroke natural.
   const weight=t<0?back:t<IMPACT_DELAY?strength+(1-strength)*smooth(t/IMPACT_DELAY):1;
   authored.action.time=t<0?1:t<IMPACT_DELAY?1+t:impactTime+Math.min(Math.max(0,t-IMPACT_DELAY),authored.duration-impactTime)*strength;
   authored.action.weight=weight;authored.address.time=0;authored.address.weight=1-weight;
   rig.mixer.update(0);scene.updateWorldMatrix(true,true);
   // Only ground contact is corrected after the baked action. Club and body
   // timing come from the same authored animation, including the impact frame.
   // Terrain correction is applied to the scene, so repeated paused frames
   // cannot accumulate an IK offset in bones owned by AnimationMixer.
   scene.position.y=feet?Math.min(...feet.current):0;
   if(grip.current){grip.current.position.copy(grip.current.parent!.worldToLocal(rig.anchor.getWorldPosition(v())));grip.current.quaternion.copy(grip.current.parent!.getWorldQuaternion(new Quaternion()).invert().multiply(rig.anchor.getWorldQuaternion(new Quaternion())));}
   return;
  }
  bones.Hips.position.copy(toWorld(v(pose.weight,.90+follow*.07,-.04)).applyMatrix4(bones.Hips.parent!.matrixWorld.clone().invert()));
  rotation('Hips',hipTurn,.12*(1-follow));rotation('Spine',turn*.6,lean);rotation('Spine2',turn,lean);rotation('Head',turn*Math.max(0,follow-.25),.31*(1-follow));
  for(const side of ['Left','Right']){
   const sign=side==='Left'?1:-1,heel=side==='Right'?pose.heel:0;
   limb(bones[side+'UpLeg'],bones[side+'Leg'],bones[side+'Foot'],toWorld(v(sign*.235,.115+heel+(feet?.current[side==='Left'?0:1]??0),.01)),toWorld(v(sign*.26,.5,.8)));
  }
  const hands=pose.hands,shaft=pose.shaft;
  for(const side of ['Left','Right']){
   const hand=bones[side+'Hand'],across=shaft.clone(),fingers=v(side==='Left'?-.75:.75,-.2,1);fingers.addScaledVector(across,-fingers.dot(across)).normalize();
   const palm=across.clone().cross(fingers).normalize(),localQ=new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(across,fingers,palm)).multiply(rig.calibration[side]);
   // Place the centre of the authored closed fingers on the shaft. A wrist
   // target alone cannot guarantee contact when hands have different bind axes.
   const goal=hands.clone().addScaledVector(shaft,side==='Left'?.055:.145).sub(rig.gripCenters[side].clone().applyQuaternion(localQ).multiplyScalar(S*.80));
   limb(bones[side+'Arm'],bones[side+'ForeArm'],hand,toWorld(goal),toWorld(v(side==='Left'?.28:-.52,1.02,side==='Left'?.36:.08)));
   const q=rootQ.clone().multiply(localQ);
   hand.quaternion.copy(hand.parent!.getWorldQuaternion(new Quaternion()).invert().multiply(q));hand.scale.multiplyScalar(.80);
   for(const finger of ['Thumb','Index','Middle','Ring','Pinky'])for(let joint=1;joint<=3;joint++){const name=side+'Hand'+finger+joint,b=bones[name];if(b&&rig.grips[name])b.quaternion.copy(rig.grips[name]);}
   hand.updateWorldMatrix(false,true);
  }
  if(grip.current){grip.current.position.copy(hands);grip.current.quaternion.copy(clubFrame(shaft,pose.faceTurn));}
 });
 return <group><primitive object={rig.scene}/>{!seated&&<group ref={grip} name="PlayerClubGrip"><GolfClub putter={club===13} iron={club>=3&&club<13}/></group>}</group>;
}
