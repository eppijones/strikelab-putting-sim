import {Suspense,useEffect,useMemo,useRef,useState,type MutableRefObject} from 'react';
import {Canvas,useFrame,useThree} from '@react-three/fiber';
import {Environment,Line,Html,useGLTF,useTexture,useEnvironment} from '@react-three/drei';
import {ACESFilmicToneMapping,Color,DataTexture,Group,Mesh,PerspectiveCamera,RGBAFormat,Vector3,type DirectionalLight} from 'three';
import {ASSETS,type Hole} from './data';
import {Ground,Woodland,Turf,TeeDetails,GROUND_TEXTURES} from './MyhraWorld';
import MyhraGolfer from './MyhraGolfer';
import {stanceDistance,IMPACT_DELAY,BALL_RADIUS} from './golfMotion';
import {GolfCart,Pond} from './MyhraFeatures';
import {bearingTo,distance,heightAt,type ShotResult,type Vec3} from '../course/engine';
import type {SwingController} from '../course/swing';
import type {TravelInput} from '../course/useDriveInput';
import {advanceCart,samplePath,treeCollisionIndex,type CartState} from './sceneMotion';
import {GreenReading} from './GreenReading';
import {useCameraInput} from './useCameraInput';

export interface PerformanceSample {fps:number;p95:number;p99:number;triangles:number;drawCalls:number}
export interface SceneProps {
 cameraReset?:number;onTravel?:(speed:number,metres:number)=>void;hole:Hole;ball:Vec3;bearing:number;result?:ShotResult;motion:number;swing:MutableRefObject<SwingController>;club:number;
 view:'player'|'overview'|'green';quality:'economy'|'balanced'|'high';onReady:()=>void;onSettled:()=>void;onAim:(p:Vec3)=>void;preview:ShotResult;
 showPrediction?:boolean;showGreenGrid?:boolean;replay?:{time:number;view:'player'|'follow'|'overhead'};onPerformance?:(sample:PerformanceSample)=>void;
 busy?:boolean;hour?:number;cartMode?:boolean;travel?:MutableRefObject<TravelInput>;drivingAllowed?:boolean;
 shotType?:'full'|'chip'|'pitch'|'putt';
 onImpact?:()=>void;
}
function FrameBudget({economy,onPerformance}:{economy:boolean;onPerformance?:SceneProps['onPerformance']}){
 const {invalidate}=useThree(),sample=useRef({since:0,times:[] as number[]});
 useEffect(()=>{if(!economy)return;const id=setInterval(invalidate,1000/30);return()=>clearInterval(id);},[economy,invalidate]);
 useFrame(({gl},delta)=>{if(!onPerformance)return;const s=sample.current;s.times.push(delta*1000);s.since+=delta;if(s.since>=1){const times=s.times.slice().sort((a,b)=>a-b);onPerformance({fps:s.times.length/s.since,p95:times[Math.floor((times.length-1)*.95)],p99:times[Math.floor((times.length-1)*.99)],triangles:gl.info.render.triangles,drawCalls:gl.info.render.calls});s.times=[];s.since=0;}});
 return null;
}
function Flag({position}:{position:Vec3}){
 const cloth=useRef<Mesh>(null);
 useFrame(({clock})=>{if(cloth.current)cloth.current.rotation.y=Math.sin(clock.elapsedTime*.7)*.12;});
 return <group position={position}><mesh rotation={[-Math.PI/2,0,0]} position={[0,.004,0]}><circleGeometry args={[.054,32]}/><meshStandardMaterial color="#0c1510" roughness={1}/></mesh><mesh position={[0,1.1,0]} castShadow><cylinderGeometry args={[.009,.012,2.2,10]}/><meshStandardMaterial color="#e9e4d7" roughness={.35} metalness={.4}/></mesh><group ref={cloth} position={[0,2.01,0]}><mesh position={[.225,0,0]}><planeGeometry args={[.45,.29,10,3]}/><meshStandardMaterial color="#e7c879" side={2} roughness={.8}/></mesh></group></group>;
}
function GolfBall({meshRef}:{meshRef:MutableRefObject<Mesh|null>}){
 const normal=useMemo(()=>{const n=256,a=new Uint8Array(n*n*4);for(let y=0;y<n;y++)for(let x=0;x<n;x++){const row=Math.floor(y/16),u=((x+(row%2)*8)%16)/16-.5,w=(y%16)/16-.5,r=Math.hypot(u,w),d=r<.38?Math.sin(r/.38*Math.PI)*.25:0;const i=(y*n+x)*4;a[i]=128+(u/(r||1))*d*127;a[i+1]=128+(w/(r||1))*d*127;a[i+2]=255;a[i+3]=255;}const t=new DataTexture(a,n,n,RGBAFormat);t.needsUpdate=true;return t;},[]);
 useEffect(()=>()=>normal.dispose(),[normal]);
 return <mesh ref={meshRef} castShadow><sphereGeometry args={[BALL_RADIUS,20,12]}/><meshPhysicalMaterial color="#fffdf3" roughness={.3} clearcoat={.4} normalMap={normal}/></mesh>;
}
function ShotTrace({result,elapsed}:{result:ShotResult;elapsed:MutableRefObject<number>}){
 const line=useRef<React.ComponentRef<typeof Line>>(null);
 const lengths=useMemo(()=>{const a=[0];for(let i=1;i<result.path.length;i++)a.push(a[i-1]+new Vector3(...result.path[i]).distanceTo(new Vector3(...result.path[i-1])));return a;},[result]);
 useFrame(()=>{if(!line.current)return;line.current.visible=elapsed.current>IMPACT_DELAY;const seconds=Math.max(0,elapsed.current-IMPACT_DELAY),times=result.pathTimes;let i=0,f=0;
  if(times?.length===result.path.length){let hi=times.length-1;while(i+1<hi){const mid=(i+hi)>>1;if(times[mid]<=seconds)i=mid;else hi=mid;}f=Math.min(1,(seconds-times[i])/Math.max(.000001,times[hi]-times[i]));}
  else {const step=Math.min(result.path.length-1,seconds/Math.max(.001,result.duration)*(result.path.length-1));i=Math.floor(step);f=step-i;}
  line.current.material.dashSize=lengths[i]+((lengths[i+1]??lengths[i])-lengths[i])*f;
 });
 return <Line ref={line} points={result.path} color="#fff0c4" lineWidth={1.8} dashed dashSize={0} gapSize={100000} transparent opacity={.72} depthWrite={false}/>;
}
function Prediction({result,hole,putting}:{result:ShotResult;hole:Hole;putting:boolean}){
 const paths=useMemo(()=>{const stride=Math.max(1,Math.floor(result.path.length/220)),air:Vec3[]=[],roll:Vec3[]=[];let landed=putting;
  for(let i=0;i<result.path.length;i+=stride){const p=result.path[i],ground=heightAt(hole.world,p[0],p[2]);if(i>stride&&p[1]-ground<.05)landed=true;(landed?roll:air).push([p[0],Math.max(p[1],ground+.026),p[2]]);}
  const p=result.path[result.path.length-1];roll.push([p[0],Math.max(p[1],heightAt(hole.world,p[0],p[2])+.026),p[2]]);if(air.length&&roll.length)air.push(roll[0]);return {air,roll};
 },[result,hole,putting]);
 return <group>{paths.air.length>1&&<Line points={paths.air} color="#f9e2a9" lineWidth={1.7} transparent opacity={.68} depthWrite={false}/>} {paths.roll.length>1&&<Line points={paths.roll} color={putting?'#ffe5a1':'#d9f1d8'} lineWidth={putting?2:1.8} dashed={!putting} dashSize={.45} gapSize={.24} transparent opacity={.84} depthWrite={false}/>}</group>;
}
interface Runtime {animation:{motion:number;started:number;done:boolean;impact:boolean;origin:Vec3;heading:number;club:number};drive:CartState}
function CourseView(p:SceneProps&{runtimeRef:MutableRefObject<Runtime>}){
 // Ready means the playing surface, golfer and lighting are downloaded,
 // rather than merely that the React canvas has mounted.
 useGLTF(ASSETS+(p.quality==='high'?'golfer-motions.glb':'golfer-motions-mobile.glb'));useTexture([...GROUND_TEXTURES,ASSETS+'fir-atlas.webp']);useEnvironment({files:ASSETS+'morning-mobile.hdr'});
 const {hole,ball,bearing,result,motion,swing,club,view,quality,onReady,onSettled,onAim,preview,hour=10,cartMode=false,travel,runtimeRef}=p;
 const high=quality==='high',economy=quality==='economy',ballMesh=useRef<Mesh>(null),golfer=useRef<Group>(null),sun=useRef<DirectionalLight>(null),elapsed=useRef(-1);
 const cart=useRef<Group>(null),feet=useRef<[number,number]>([0,0]),look=useRef(new Vector3()),initialized=useRef(false);
 const orbit=useCameraInput(!p.busy&&!cartMode&&!p.replay,p.cameraReset??0),trees=useMemo(()=>treeCollisionIndex(hole.data.trees),[hole]);
 const vectors=useRef({direction:new Vector3(),right:new Vector3(),target:new Vector3(),position:new Vector3(),offset:new Vector3(),up:new Vector3(0,1,0)});
 const [anchor,setAnchor]=useState<Vec3>(ball),lastReport=useRef(0);
 const sunAngle=(hour-6)/14*Math.PI,solar=Math.max(.1,Math.sin(sunAngle)),sunColor=new Color().setHSL(.105,.12+(1-solar)*.5,.93);
 const putting=club===13,reading=(putting||view==='green')&&(p.showGreenGrid??true)&&!cartMode&&!p.busy&&!p.replay;
 const wasDriving=useRef(false);
 useEffect(()=>{onReady();},[onReady]);
 useFrame(({camera,size},delta)=>{
  const {direction,right,target,position,offset,up}=vectors.current,dt=Math.min(delta,.1),r=runtimeRef.current.animation;
  if(r.motion!==motion){r.motion=motion;r.started=performance.now()/1000;r.done=!result;r.impact=false;r.origin=result?[...result.path[0]]:[...ball];r.heading=result&&result.path.length>1?bearingTo(result.path[0],result.path[Math.min(5,result.path.length-1)]):bearing;r.club=club;}
  let visual:Vec3=[ball[0],heightAt(hole.world,ball[0],ball[2])+BALL_RADIUS,ball[2]];
  if(result&&(!r.done||p.replay)){const time=p.replay?.time??performance.now()/1000-r.started;if(!p.replay&&!r.impact&&time>=IMPACT_DELAY){r.impact=true;p.onImpact?.();}visual=samplePath(result.path,time-IMPACT_DELAY,result.duration,result.pathTimes);elapsed.current=time;if(!p.replay&&time>=result.duration+IMPACT_DELAY){r.done=true;elapsed.current=-1;onSettled();setAnchor(ball);}}
  else {elapsed.current=-1;if(!cartMode&&distance(anchor,ball)>3)setAnchor(ball);}
  const animating=!!result&&(!r.done||!!p.replay),origin=p.replay&&result?result.path[0]:animating?r.origin:ball,heading=p.replay?bearing:animating?r.heading:bearing,groundY=heightAt(hole.world,origin[0],origin[2]);
  direction.set(Math.sin(heading),0,-Math.cos(heading));right.set(Math.cos(heading),0,Math.sin(heading));
  // Shot paths contain sphere centres already; never add a second ball radius.
  ballMesh.current?.position.set(...visual);
  if(golfer.current){const gx=origin[0]-right.x*stanceDistance(club),gz=origin[2]-right.z*stanceDistance(club);golfer.current.position.set(gx,groundY,gz);feet.current=[heightAt(hole.world,gx+direction.x*.235,gz+direction.z*.235)-groundY,heightAt(hole.world,gx-direction.x*.235,gz-direction.z*.235)-groundY];golfer.current.rotation.y=Math.PI/2-heading;golfer.current.visible=!cartMode&&view!=='overview'&&(!animating||elapsed.current<3||p.replay?.view==='player');}
  const car=runtimeRef.current.drive;
  if(cartMode&&!wasDriving.current){const road=hole.data.road;let index=0;for(let i=1;i<road.length;i++)if(distance(road[i],ball)<distance(road[index],ball))index=i;car.x=road[index][0];car.z=road[index][2];car.heading=bearingTo(road[index],road[Math.min(road.length-1,index+3)]??road[index]);car.speed=0;car.steer=0;}
  wasDriving.current=cartMode;
  advanceCart(car,travel?.current,dt,hole.world,trees,!!p.drivingAllowed&&!p.replay);
  if(cartMode&&distance(anchor,[car.x,0,car.z])>12)setAnchor([car.x,heightAt(hole.world,car.x,car.z),car.z]);
  if(cartMode&&performance.now()-lastReport.current>200){lastReport.current=performance.now();p.onTravel?.(Math.abs(car.speed),distance([car.x,0,car.z],ball));}
  if(cart.current){const fx=Math.sin(car.heading),fz=-Math.cos(car.heading),rx=-Math.cos(car.heading),rz=-Math.sin(car.heading),front=heightAt(hole.world,car.x+fx*.85,car.z+fz*.85),back=heightAt(hole.world,car.x-fx*.85,car.z-fz*.85),left=heightAt(hole.world,car.x+rx*.6,car.z+rz*.6),rightH=heightAt(hole.world,car.x-rx*.6,car.z-rz*.6);cart.current.position.set(car.x,(front+back+left+rightH)/4,car.z);cart.current.rotation.set(-Math.atan2(front-back,1.7),Math.PI-car.heading,Math.atan2(left-rightH,1.2),'YXZ');}
  if(sun.current){const x=cartMode?car.x:origin[0],z=cartMode?car.z:origin[2],y=heightAt(hole.world,x,z);sun.current.position.set(x-Math.cos(sunAngle)*75,y+25+solar*80,z+45);sun.current.target.position.set(x,y,z);sun.current.target.updateMatrixWorld();}
  const portrait=size.width<size.height,mode=p.replay?.view==='overhead'?'overview':view;
  if(cartMode){target.set(car.x+Math.sin(car.heading)*2,heightAt(hole.world,car.x,car.z)+1.2,car.z-Math.cos(car.heading)*2);position.set(car.x-Math.sin(car.heading)*6.7,target.y+2.6,car.z+Math.cos(car.heading)*6.7);}
  else if(mode==='overview'){target.set(384,5,390);position.set(435,175,560);}
  else if(animating&&putting&&result&&distance(origin,result.end)<12){const finish=p.replay?result.end:hole.data.pin;target.set((origin[0]+finish[0])*.5,heightAt(hole.world,origin[0],origin[2])+.2,(origin[2]+finish[2])*.5);position.copy(target).addScaledVector(direction,-Math.max(4,distance(origin,finish)*.65));position.y+=3.6;}
  else if(animating&&elapsed.current>(putting?2.2:1.15)&&p.replay?.view!=='player'){target.set(visual[0],visual[1]+.15,visual[2]);position.copy(target).addScaledVector(direction,-(putting?4.5:10)).addScaledVector(right,putting?0:2.1);position.y+=putting?2.8:5;}
  else {const green=mode==='green',back=green?7:portrait?5.7:putting?4.1:4.8;target.set(origin[0],groundY+(green?.2:portrait?.83:.65),origin[2]).addScaledVector(direction,green?Math.min(5,distance(ball,hole.data.pin)*.45):putting?1.8:3.2).addScaledVector(right,portrait?-.42:0);position.set(origin[0],groundY+(green?6:putting?2.25:portrait?2.65:2.2),origin[2]).addScaledVector(direction,-back).addScaledVector(right,portrait?-.6:-.85);}
  if(!cartMode&&!animating){offset.copy(position).sub(target).applyAxisAngle(up,orbit.current.yaw);offset.y+=orbit.current.pitch*offset.length();offset.multiplyScalar(orbit.current.zoom);position.copy(target).add(offset);}
  // Keep the complete sight line clear of hills, not just the camera endpoint.
  position.y=Math.max(position.y,heightAt(hole.world,position.x,position.z)+1.1);
  for(let i=1;i<9;i++){const f=i/10,x=target.x+(position.x-target.x)*f,z=target.z+(position.z-target.z)*f,clear=heightAt(hole.world,x,z)+.35;if(target.y+(position.y-target.y)*f<clear)position.y=Math.max(position.y,target.y+(clear-target.y)/f);}
  if(!initialized.current){camera.position.copy(position);look.current.copy(target);initialized.current=true;}else{camera.position.lerp(position,1-Math.exp(-dt*(cartMode?4.8:4)));look.current.lerp(target,1-Math.exp(-dt*(cartMode?6:5)));}
  camera.lookAt(look.current);if(camera instanceof PerspectiveCamera){camera.fov+=((mode==='green'?46:49)-camera.fov)*(1-Math.exp(-dt*4));camera.updateProjectionMatrix();}
 });
 return <>
  <FrameBudget economy={economy} onPerformance={p.onPerformance}/><Environment files={ASSETS+'morning-mobile.hdr'} background backgroundIntensity={.45+solar*.65} environmentIntensity={.25+solar*.35} backgroundRotation={[0,1.8,0]} environmentRotation={[0,1.8,0]}/>
  <fog attach="fog" args={['#b7c9ce',220,820]}/><hemisphereLight args={['#d4e8f2','#465334',.68]}/>
  <directionalLight key={quality} ref={sun} color={sunColor} intensity={2.8*solar+.45} castShadow shadow-mapSize={economy?[1024,1024]:[2048,2048]} shadow-camera-left={-24} shadow-camera-right={24} shadow-camera-top={24} shadow-camera-bottom={-24} shadow-camera-near={1} shadow-camera-far={220} shadow-bias={-.00008} shadow-normalBias={.018}/>
  <Ground hole={hole} onAim={onAim} anchor={anchor} cameraInput={orbit}/><Woodland hole={hole} anchor={anchor} high={high}/><Turf hole={hole} anchor={anchor} high={high}/><TeeDetails hole={hole}/><Pond hole={hole} high={high}/><Suspense fallback={null}><GolfCart actor={cart} high={high}>{cartMode&&<group position={[.30,0,-.16]}><MyhraGolfer swing={swing} elapsed={elapsed} club={13} high={high} seated/></group>}</GolfCart></Suspense>
  <Flag position={hole.data.pin}/><GolfBall meshRef={ballMesh}/><group ref={golfer}><MyhraGolfer swing={swing} elapsed={elapsed} club={club} high={high} feet={feet} replay={!!p.replay} shotType={p.shotType}/></group>
  {result&&<ShotTrace result={result} elapsed={elapsed}/>}{reading&&<GreenReading hole={hole} ball={ball}/>}
  {!p.busy&&!cartMode&&!p.replay&&(p.showPrediction??true)&&<Prediction result={preview} hole={hole} putting={putting}/>}
  {!p.busy&&!cartMode&&!p.replay&&<group position={[preview.end[0],heightAt(hole.world,preview.end[0],preview.end[2])+.03,preview.end[2]]}><mesh rotation={[-Math.PI/2,0,0]}><ringGeometry args={[Math.max(.09,Math.min(3.6,distance(ball,preview.end)*.018)),Math.max(.12,Math.min(4,distance(ball,preview.end)*.020)),48]}/><meshBasicMaterial color="#f2d18b" transparent opacity={.8} depthWrite={false}/></mesh>{distance(ball,preview.end)>10&&<Html center position={[0,view==='overview'?6:2.5,0]}><span className="mh-target-label">{Math.round(distance(ball,preview.end))} m<span>PROJECTED FINISH</span></span></Html>}</group>}
 </>;
}
export default function MyhraScene(props:SceneProps){
 const [surfaceKey,setSurfaceKey]=useState(0),runtime=useRef<Runtime>({animation:{motion:props.motion,started:0,done:true,impact:false,origin:[...props.ball],heading:props.bearing,club:props.club},drive:{x:props.hole.data.road[25][0],z:props.hole.data.road[25][2],heading:bearingTo(props.hole.data.road[25],props.hole.data.road[30]),speed:0,steer:0}});
 useEffect(()=>{if(!/AppleWebKit/.test(navigator.userAgent)||/Chrome|Chromium|Edg|OPR/.test(navigator.userAgent))return;const orientation=matchMedia('(orientation: portrait)'),reset=()=>setSurfaceKey(k=>k+1);orientation.addEventListener('change',reset);return()=>orientation.removeEventListener('change',reset);},[]);
 return <Canvas key={surfaceKey} shadows frameloop={props.quality==='economy'?'demand':'always'} dpr={props.quality==='high'?[1,1.5]:props.quality==='economy'?[.8,1]:[1,1.2]} camera={{fov:49,near:.04,far:1000,position:[384,17,576]}} gl={{antialias:true,alpha:false,powerPreference:'high-performance',toneMapping:ACESFilmicToneMapping,toneMappingExposure:.91}}><Suspense fallback={null}><CourseView {...props} runtimeRef={runtime}/></Suspense></Canvas>;
}
