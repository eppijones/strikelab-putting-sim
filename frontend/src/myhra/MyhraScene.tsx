import {Suspense,useEffect,useMemo,useRef,useState,type MutableRefObject} from 'react';
import {Canvas,useFrame} from '@react-three/fiber';
import {Environment,Line,OrbitControls,Html} from '@react-three/drei';
import {ACESFilmicToneMapping,Color,DataTexture,Group,Mesh,RGBAFormat,Vector3,type DirectionalLight} from 'three';
import {ASSETS,type Hole} from './data';
import {Ground,Woodland,Turf,TeeDetails} from './MyhraWorld';
import MyhraGolfer from './MyhraGolfer';
import {stanceDistance,IMPACT_DELAY} from './golfMotion';
import {GolfCart,Pond} from './MyhraFeatures';
import {bearingTo,distance,heightAt,surfaceAt,type ShotResult,type Vec3} from '../course/engine';
import type {SwingController} from '../course/swing';

import type {TravelInput} from '../course/useDriveInput';
export interface SceneProps {cameraReset?:number;onTravel?:(speed:number,metres:number)=>void;hole:Hole;ball:Vec3;bearing:number;result?:ShotResult;motion:number;swing:MutableRefObject<SwingController>;club:number;view:'player'|'overview'|'green';quality:'balanced'|'high';onReady:()=>void;onSettled:()=>void;onAim:(p:Vec3)=>void;preview:Vec3;busy?:boolean;hour?:number;cartMode?:boolean;travel?:MutableRefObject<TravelInput>;drivingAllowed?:boolean}

function Flag({position}:{position:Vec3}){
 const cloth=useRef<Mesh>(null);
 useFrame(({clock})=>{if(cloth.current)cloth.current.rotation.y=Math.sin(clock.elapsedTime*.7)*.12;});
 return <group position={position}>
  <mesh rotation={[-Math.PI/2,0,0]} position={[0,.004,0]}><circleGeometry args={[.054,32]}/><meshStandardMaterial color="#0c1510" roughness={1}/></mesh>
  <mesh position={[0,1.1,0]} castShadow><cylinderGeometry args={[.009,.012,2.2,10]}/><meshStandardMaterial color="#e9e4d7" roughness={.35} metalness={.4}/></mesh>
  <group ref={cloth} position={[0,2.01,0]}><mesh position={[.225,0,0]}><planeGeometry args={[.45,.29,10,3]}/><meshStandardMaterial color="#d8b667" side={2} roughness={.8}/></mesh></group>
 </group>;
}
function GreenGrid({hole,ball}:{hole:Hole;ball:Vec3}){
 const lines=useMemo(()=>{const a:Vec3[][]=[];for(let i=-9;i<=9;i++){const row:Vec3[]=[],col:Vec3[]=[];for(let j=-9;j<=9;j++){row.push([ball[0]+i,heightAt(hole.world,ball[0]+i,ball[2]+j)+.02,ball[2]+j]);col.push([ball[0]+j,heightAt(hole.world,ball[0]+j,ball[2]+i)+.02,ball[2]+i]);}a.push(row,col);}return a;},[hole,ball]);
 return <group>{lines.map((p,i)=><Line key={i} points={p} color="#edf6c7" transparent opacity={.22} lineWidth={.5}/>)}</group>;
}
function GolfBall({meshRef}:{meshRef:MutableRefObject<Mesh|null>}){
 const normal=useMemo(()=>{const n=512,a=new Uint8Array(n*n*4);for(let y=0;y<n;y++)for(let x=0;x<n;x++){const row=Math.floor(y/32),u=((x+(row%2)*16)%32)/32-.5,w=(y%32)/32-.5,r=Math.hypot(u,w),d=r<.38?Math.sin(r/.38*Math.PI)*.25:0;const i=(y*n+x)*4;a[i]=128+(u/(r||1))*d*127;a[i+1]=128+(w/(r||1))*d*127;a[i+2]=255;a[i+3]=255;}const t=new DataTexture(a,n,n,RGBAFormat);t.needsUpdate=true;return t;},[]);
 return <mesh ref={meshRef} castShadow><sphereGeometry args={[.02135,24,16]}/><meshPhysicalMaterial color="#fffdf3" roughness={.3} clearcoat={.4} normalMap={normal}/></mesh>;
}
function ShotTrace({result,elapsed}:{result:ShotResult;elapsed:MutableRefObject<number>}){
 const line=useRef<React.ComponentRef<typeof Line>>(null);
 const lengths=useMemo(()=>{const a=[0];for(let i=1;i<result.path.length;i++)a.push(a[i-1]+new Vector3(...result.path[i]).distanceTo(new Vector3(...result.path[i-1])));return a;},[result]);
 useFrame(()=>{if(!line.current)return;line.current.visible=elapsed.current>IMPACT_DELAY;
  const step=Math.min(result.path.length-1,Math.max(0,elapsed.current-IMPACT_DELAY)/Math.min(12,Math.max(1.4,result.duration))*(result.path.length-1)),i=Math.floor(step),f=step-i;
  line.current.material.dashSize=lengths[i]+((lengths[i+1]??lengths[i])-lengths[i])*f;
 });
 return <Line ref={line} points={result.path} color="#fff0c4" lineWidth={1.35} dashed dashSize={0} gapSize={100000} transparent opacity={.55} depthWrite={false}/>;
}
interface Runtime {animation:{motion:number;started:number;done:boolean;origin:Vec3;heading:number;club:number};drive:{x:number;z:number;heading:number;speed:number}}
function CourseView(p:SceneProps&{runtimeRef:MutableRefObject<Runtime>}){
 const {hole,ball,bearing,result,motion,swing,club,view,quality,onReady,onSettled,onAim,preview,hour=10,cartMode=false,travel,runtimeRef}=p;
 const high=quality==='high',ballMesh=useRef<Mesh>(null),golfer=useRef<Group>(null),sun=useRef<DirectionalLight>(null),elapsed=useRef(-1);
 const controls=useRef<React.ComponentRef<typeof OrbitControls>>(null);
 const cart=useRef<Group>(null),feet=useRef<[number,number]>([0,0]);
 const [anchor,setAnchor]=useState<Vec3>(ball),lastKey=useRef(''),lastReport=useRef(0);
 const sunAngle=(hour-6)/14*Math.PI,solar=Math.max(.1,Math.sin(sunAngle)),sunColor=new Color().setHSL(.105,.12+(1-solar)*.5,.93);
 useEffect(()=>{onReady();},[onReady]);
 useFrame(({camera,size},delta)=>{
  const direction=new Vector3(),target=new Vector3(),position=new Vector3();
  const dt=Math.min(delta,.05),r=runtimeRef.current.animation;
  if(r.motion!==motion){r.motion=motion;r.started=performance.now()/1000;r.done=!result;r.origin=result?[...result.path[0]]:[...ball];r.heading=result&&result.path.length>1?bearingTo(result.path[0],result.path[Math.min(5,result.path.length-1)]):bearing;r.club=club;}
  let visual=ball;
  if(result&&!r.done){
   const time=performance.now()/1000-r.started,duration=Math.min(12,Math.max(1.4,result.duration)),step=Math.min(result.path.length-1,Math.max(0,time-IMPACT_DELAY)/duration*(result.path.length-1));
   const i=Math.floor(step),a=result.path[i],b=result.path[Math.min(i+1,result.path.length-1)],f=step-i;
   visual=[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];elapsed.current=time;
   if(step>=result.path.length-1){r.done=true;elapsed.current=-1;onSettled();setAnchor(ball);}
  }else {elapsed.current=-1;if(!cartMode&&distance(anchor,ball)>3)setAnchor(ball);}
  const origin=r.done?ball:r.origin,heading=r.done?bearing:r.heading;
  direction.set(Math.sin(heading),0,-Math.cos(heading));const right=new Vector3(Math.cos(heading),0,Math.sin(heading));
  ballMesh.current?.position.set(visual[0],visual[1]+.024+(distance((r.done?ball:r.origin),hole.data.tee)<.5&&(!result||r.done||elapsed.current<IMPACT_DELAY)?.03:0),visual[2]);
  if(golfer.current){const gx=origin[0]-right.x*stanceDistance(club),gz=origin[2]-right.z*stanceDistance(club);golfer.current.position.set(gx,origin[1],gz);feet.current=[heightAt(hole.world,gx+direction.x*.235,gz+direction.z*.235)-origin[1],heightAt(hole.world,gx-direction.x*.235,gz-direction.z*.235)-origin[1]];golfer.current.rotation.y=Math.PI/2-heading;golfer.current.visible=!cartMode&&view!=='overview'&&(r.done||elapsed.current<3);}
  if(sun.current){sun.current.position.set(origin[0]-Math.cos(sunAngle)*75,origin[1]+25+solar*80,origin[2]+45);sun.current.target.position.set(...origin);sun.current.target.updateMatrixWorld();}
  const car=runtimeRef.current.drive,input=travel?.current;
  car.speed+=(p.drivingAllowed?(input?.forward??0)*(input?.boost?22:8)-car.speed:-car.speed)*(1-Math.exp(-Math.min(delta,.1)*(Math.abs(input?.forward??0)<.01?9:3)));
  if(p.drivingAllowed){car.heading+=(input?.turn??0)*dt*1.25*Math.min(1,Math.abs(car.speed)/2)*(car.speed<0?-1:1);const steps=Math.max(1,Math.ceil(Math.abs(car.speed)*dt/.15));for(let i=0;i<steps;i++){const x=car.x+Math.sin(car.heading)*car.speed*dt/steps,z=car.z-Math.cos(car.heading)*car.speed*dt/steps,l=surfaceAt(hole.world,x,z);if(l!=='water'&&l!=='out'&&l!=='green'&&l!=='bunker'&&Math.abs(heightAt(hole.world,x,z)-heightAt(hole.world,car.x,car.z))<.3){car.x=x;car.z=z;}else{car.speed=0;break;}}if(distance(anchor,[car.x,0,car.z])>15)setAnchor([car.x,heightAt(hole.world,car.x,car.z),car.z]);}
  if(cartMode&&performance.now()-lastReport.current>200){lastReport.current=performance.now();p.onTravel?.(Math.abs(car.speed),distance([car.x,0,car.z],ball));}
  if(cart.current){
   const fx=Math.sin(car.heading),fz=-Math.cos(car.heading),rx=-Math.cos(car.heading),rz=-Math.sin(car.heading);
   const front=heightAt(hole.world,car.x+fx*.85,car.z+fz*.85),back=heightAt(hole.world,car.x-fx*.85,car.z-fz*.85),left=heightAt(hole.world,car.x+rx*.6,car.z+rz*.6),rightH=heightAt(hole.world,car.x-rx*.6,car.z-rz*.6);
   cart.current.position.set(car.x,(front+back+left+rightH)/4,car.z);cart.current.rotation.set(-Math.atan2(front-back,1.7),Math.PI-car.heading,Math.atan2(left-rightH,1.2),'YXZ');
   if(cartMode&&sun.current){sun.current.position.set(car.x-Math.cos(sunAngle)*75,cart.current.position.y+25+solar*80,car.z+45);sun.current.target.position.copy(cart.current.position);sun.current.target.updateMatrixWorld();}
  }
  const key=`${p.cameraReset}:${motion}:${r.done}:${view}:${cartMode}:${ball.join(',')}:${size.width<700}:${heading.toFixed(5)}`;
  if(cartMode){target.set(car.x,heightAt(hole.world,car.x,car.z)+1.15,car.z);position.copy(target).add(new Vector3(-Math.sin(car.heading)*6.5,2.4,Math.cos(car.heading)*6.5));if(key!==lastKey.current){camera.position.copy(position);controls.current?.target.copy(target);camera.lookAt(target);}else{camera.position.lerp(position,1-Math.exp(-dt*4));controls.current?.target.lerp(target,1-Math.exp(-dt*6));}}
  else if(view==='overview'){
   if(key!==lastKey.current){target.set(393,7,385);position.set(453,170,605);controls.current?.target.copy(target);camera.position.copy(position);camera.lookAt(target);}
  }else if(!r.done&&elapsed.current>1.15){
   target.set(visual[0],visual[1]+.25,visual[2]);position.copy(target).addScaledVector(direction,-8).addScaledVector(right,2.2);position.setY(Math.max(visual[1]+4,heightAt(hole.world,position.x,position.z)+3));
   camera.position.lerp(position,1-Math.exp(-dt*2.7));controls.current?.target.lerp(target,1-Math.exp(-dt*4));
  }else{
   const portrait=size.width<size.height,putting=club===13,back=view==='green'?8:portrait?5.9:putting?4.5:4.7;
   target.set(origin[0],origin[1]+(portrait?.95:.8),origin[2]).addScaledVector(direction,view==='green'?2:3.3).addScaledVector(right,portrait?-.6:0);
   position.set(...origin).addScaledVector(direction,-back).addScaledVector(right,portrait?-.65:-.95);position.setY(origin[1]+(view==='green'?6.5:portrait?2.7:2.15));
   if(key!==lastKey.current){camera.position.copy(position);controls.current?.target.copy(target);camera.lookAt(target);}
  }
  lastKey.current=key;
 });
 return <>
  <Environment files={ASSETS+'morning.hdr'} background backgroundIntensity={.45+solar*.65} environmentIntensity={.25+solar*.35} backgroundRotation={[0,1.8,0]} environmentRotation={[0,1.8,0]}/>
  <fog attach="fog" args={['#b7c9ce',180,760]}/><hemisphereLight args={['#d4e8f2','#465334',.8]}/>
  <directionalLight key={quality} ref={sun} color={sunColor} intensity={2.8*solar+.45} castShadow shadow-mapSize={high?[4096,4096]:[2048,2048]} shadow-camera-left={-16} shadow-camera-right={16} shadow-camera-top={16} shadow-camera-bottom={-16} shadow-camera-near={1} shadow-camera-far={220} shadow-bias={-.00008} shadow-normalBias={.018}/>
  <Ground hole={hole} onAim={onAim}/><Woodland hole={hole} anchor={anchor} high={high}/><Turf hole={hole} anchor={anchor} high={high}/><TeeDetails hole={hole}/><Pond hole={hole} high={high}/><GolfCart actor={cart}>{cartMode&&<group position={[.30,0,-.16]}><MyhraGolfer swing={swing} elapsed={elapsed} club={13} seated/></group>}</GolfCart>
  <mesh position={[hole.data.tee[0],hole.data.tee[1]+.020,hole.data.tee[2]]}><cylinderGeometry args={[.005,.002,.04,8]}/><meshStandardMaterial color="#dac8a7" roughness={.7}/></mesh><Flag position={hole.data.pin}/><GolfBall meshRef={ballMesh}/><group ref={golfer}><MyhraGolfer swing={swing} elapsed={elapsed} club={club} feet={feet}/></group>
  {result&&<ShotTrace result={result} elapsed={elapsed}/>}
  {view==='green'&&<GreenGrid hole={hole} ball={ball}/>}
  {!p.busy&&!cartMode&&<group position={[preview[0],heightAt(hole.world,preview[0],preview[2])+.08,preview[2]]}><mesh rotation={[-Math.PI/2,0,0]}><ringGeometry args={[Math.min(3.6,distance(ball,preview)*.02),Math.min(4,distance(ball,preview)*.022),48]}/><meshBasicMaterial color="#f2d18b" transparent opacity={.7} depthWrite={false}/></mesh>{distance(ball,preview)>10&&<Html center position={[0,view==='overview'?6:2.5,0]}><span className="mh-target-label">{Math.round(distance(ball,preview))} m<span>YOUR TARGET</span></span></Html>}</group>}
  <OrbitControls ref={controls} enabled={!p.busy&&!cartMode} enableDamping={false} enablePan={view==='overview'} enableZoom={view!=='player'} enableRotate={!cartMode} minDistance={2} maxDistance={300} maxPolarAngle={Math.PI*.485} makeDefault/>
 </>;
}
export default function MyhraScene(props:SceneProps){
 const [surfaceKey,setSurfaceKey]=useState(0);
 const runtime=useRef<Runtime>({animation:{motion:props.motion,started:0,done:true,origin:[...props.ball],heading:props.bearing,club:props.club},drive:{x:props.hole.data.road[25][0],z:props.hole.data.road[25][2],heading:bearingTo(props.hole.data.road[25],props.hole.data.road[30]),speed:0}});
 useEffect(()=>{
  if(!/AppleWebKit/.test(navigator.userAgent)||/Chrome|Chromium|Edg|OPR/.test(navigator.userAgent))return;
  const orientation=matchMedia('(orientation: portrait)'),reset=()=>setSurfaceKey(k=>k+1);
  orientation.addEventListener('change',reset);return()=>orientation.removeEventListener('change',reset);
 },[]);
 return <Canvas key={surfaceKey} shadows dpr={props.quality==='high'?[1,1.5]:[1,1.2]} camera={{fov:49,near:.05,far:1100,position:[384,17,576]}} gl={{antialias:true,alpha:false,powerPreference:'high-performance',toneMapping:ACESFilmicToneMapping,toneMappingExposure:.88}}><Suspense fallback={null}><CourseView {...props} runtimeRef={runtime}/></Suspense></Canvas>;
}
