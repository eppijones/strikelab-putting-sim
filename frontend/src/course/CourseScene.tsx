import {memo,useEffect,useMemo,useRef,useState} from 'react';
import {Canvas,useFrame} from '@react-three/fiber';
import {OrbitControls,Sky,Line,RoundedBox} from '@react-three/drei';
import {ACESFilmicToneMapping,Vector3,type Group,type Mesh,type DirectionalLight} from 'three';
import {heightAt,distance,bearingTo,type Course,type Vec3,type World,type ShotResult} from './engine';
import {Terrain,Trees,Grass} from './Environment';
import Golfer,{type Pose} from './Golfer';
import type {SwingController} from './swing';

export type TravelMode='golf'|'walk'|'cart';
export interface Controls {forward:number;turn:number;gamepad:boolean}
interface Props {course:Course;world:World;ball:Vec3;bearing:number;result?:ShotResult;motion:number;mode:TravelMode;input:React.RefObject<Controls>;onSettled:()=>void;onReady:()=>void;onTravelDistance:(metres:number)=>void;quality:'balanced'|'high';swing:React.RefObject<SwingController>;club:number;cameraMode:'player'|'scout';preview:Vec3;journey?:Vec3;showGrid:boolean}
interface Runtime {animation:{motion:number;time:number;done:boolean;start:Vec3};traveler:{position:Vector3;heading:number}}

function Cart({actor}:{actor:React.RefObject<Group|null>}){
 return <group ref={actor}>
  <RoundedBox args={[1.3,.35,2.2]} radius={.12} position={[0,.65,0]} castShadow><meshStandardMaterial color="#eef0dc" roughness={.35} metalness={.18}/></RoundedBox>
  <RoundedBox args={[1.25,.65,.6]} radius={.08} position={[0,.92,-.9]} castShadow><meshStandardMaterial color="#233d35" roughness={.4}/></RoundedBox>
  <RoundedBox args={[1.25,.17,.75]} radius={.06} position={[0,.97,.15]}><meshStandardMaterial color="#d0bd91"/></RoundedBox>
  <RoundedBox args={[1.25,.55,.12]} radius={.04} position={[0,1.22,.5]}><meshStandardMaterial color="#d0bd91"/></RoundedBox>
  <RoundedBox args={[1.6,.1,2.5]} radius={.06} position={[0,2.02,0]} castShadow><meshStandardMaterial color="#eeeadb"/></RoundedBox>
  <mesh position={[0,1.52,-.81]} rotation={[-.12,0,0]}><boxGeometry args={[1.23,.75,.018]}/><meshPhysicalMaterial color="#badce0" transparent opacity={.22} roughness={.1}/></mesh>
  <mesh position={[-.38,1.14,-.32]} rotation={[.55,0,0]}><torusGeometry args={[.2,.022,8,20]}/><meshStandardMaterial color="#1c2524"/></mesh>
  {[-.65,.65].flatMap(x=>[-.77,.77].map(z=><group key={`${x},${z}`}><mesh position={[x,.35,z]} rotation={[0,0,Math.PI/2]} castShadow><cylinderGeometry args={[.31,.31,.2,20]}/><meshStandardMaterial color="#18201e" roughness={.9}/></mesh><mesh position={[x*1.02,.35,z]} rotation={[0,0,Math.PI/2]}><cylinderGeometry args={[.17,.17,.21,12]}/><meshStandardMaterial color="#b3bcbb" metalness={.7} roughness={.25}/></mesh><mesh position={[x,1.43,z]}><cylinderGeometry args={[.025,.025,1.1,8]}/><meshStandardMaterial color="#343e3a"/></mesh></group>))}
 </group>;
}

function GreenGrid({world,ball}:{world:World;ball:Vec3}){
 const lines=useMemo(()=>{const data:Vec3[][]=[];for(let i=-8;i<=8;i++){const a:Vec3[]=[],b:Vec3[]=[];for(let j=-8;j<=8;j++){let x=ball[0]+i,z=ball[2]+j;a.push([x,heightAt(world,x,z)+.035,z]);x=ball[0]+j;z=ball[2]+i;b.push([x,heightAt(world,x,z)+.035,z]);}data.push(a,b);}return data;},[world,ball]);
 return <group>{lines.map((l,i)=><Line key={i} points={l} color="#e8efc4" transparent opacity={.18} lineWidth={.6}/>)}</group>;
}

function GameView(props:Props&{runtimeRef:React.RefObject<Runtime>}){
 const {course,world,ball,bearing,result,motion,mode,input,onSettled,onReady,quality,swing,club,cameraMode,preview,journey,showGrid,runtimeRef}=props;
 useEffect(()=>onReady(),[onReady]);
 const ballMesh=useRef<Mesh>(null),golfer=useRef<Group>(null),cart=useRef<Group>(null),sun=useRef<DirectionalLight>(null);
 const controls=useRef<React.ComponentRef<typeof OrbitControls>>(null);
 const pose=useRef<Pose>({walking:0,time:0,swing:swing.current,follow:0,putting:club===13});
 const lastMode=useRef<TravelMode>(mode),lastKey=useRef(''),lastJourney=useRef<Vec3|undefined>(undefined);
 const [anchor,setAnchor]=useState<Vec3>(()=>[...ball]);
 const lastTravelReport=useRef(-1);
 useFrame(({camera,size},delta)=>{
  const dt=Math.min(delta,.05),anim=runtimeRef.current.animation;
  if(anim.motion!==motion){anim.motion=motion;anim.time=0;anim.done=!result;anim.start=result?[...result.path[0]]:[...ball];}
  let visual:Vec3=ball;
  if(result&&!anim.done){
   anim.time+=dt;const duration=Math.min(13,Math.max(1.4,result.duration)),progress=Math.max(0,anim.time-.28)/duration;
   const step=Math.min(result.path.length-1,progress*(result.path.length-1)),i=Math.floor(step),a=result.path[i],b=result.path[Math.min(i+1,result.path.length-1)],f=step-i;
   visual=[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];
   if(step>=result.path.length-1){anim.done=true;onSettled();}
  }
  ballMesh.current?.position.set(visual[0],visual[1]+.03,visual[2]);
  pose.current.time+=dt;pose.current.swing=swing.current;pose.current.putting=club===13;pose.current.follow=!anim.done?Math.min(1,anim.time*2.5):0;
  const t=runtimeRef.current.traveler;
  if(mode!=='golf'){
   if(lastMode.current==='golf'){
    const start=journey&&lastJourney.current!==journey?journey:ball;t.position.set(...start);t.heading=distance(start,ball)>3?bearingTo(start,ball):bearing;lastJourney.current=journey;
   }
   const speed=mode==='cart'?11:3.4;t.heading+=input.current.turn*dt*(mode==='cart'?1.2:1.8);
   const x=t.position.x+Math.sin(t.heading)*input.current.forward*dt*speed,z=t.position.z-Math.cos(t.heading)*input.current.forward*dt*speed,extent=(world.terrain.size-1)*world.terrain.spacing;
   if(x>2&&z>2&&x<extent-2&&z<extent-2){const y=heightAt(world,x,z);if(Math.abs(y-t.position.y)<1.5)t.position.set(x,y,z);}
   if(cart.current){cart.current.visible=mode==='cart';cart.current.position.copy(t.position);cart.current.rotation.y=-t.heading;}
   if(golfer.current){golfer.current.visible=mode==='walk';golfer.current.position.copy(t.position);golfer.current.rotation.y=Math.PI-t.heading;}
   pose.current.walking=Math.abs(input.current.forward);
   if(pose.current.time-lastTravelReport.current>.5){props.onTravelDistance(Math.hypot(t.position.x-ball[0],t.position.z-ball[2]));lastTravelReport.current=pose.current.time;}
   const behind=mode==='cart'?6.5:3.8,desired=new Vector3(t.position.x-Math.sin(t.heading)*behind,t.position.y+(mode==='cart'?3:2.25),t.position.z+Math.cos(t.heading)*behind);
   desired.y=Math.max(desired.y,heightAt(world,desired.x,desired.z)+.65);camera.position.lerp(desired,1-Math.exp(-dt*6));camera.lookAt(t.position.x+Math.sin(t.heading)*2,t.position.y+1.25,t.position.z-Math.cos(t.heading)*2);
  }else{
   pose.current.walking=0;if(cart.current)cart.current.visible=false;
   const start=!anim.done?anim.start:ball;
   if(golfer.current){golfer.current.visible=true;const x=start[0]-Math.cos(bearing)*.72,z=start[2]-Math.sin(bearing)*.72;golfer.current.position.set(x,heightAt(world,x,z),z);golfer.current.rotation.y=Math.PI/2-bearing;}
   const narrow=size.width/size.height<.8,offset=narrow?-.4:1.3;
   const key=ball.join(',')+motion+cameraMode+Math.round(bearing*100)+(anim.done?'rest':'shot')+narrow+(cameraMode==='scout'?preview.map(Math.round).join(','):'');
   if(lastKey.current!==key||lastMode.current!=='golf'){
    const scout=cameraMode==='scout',focus=scout?preview:start;
    const back=scout?25:narrow?6.5:4.8;
    camera.position.set(focus[0]-Math.sin(bearing)*back+Math.cos(bearing)*(scout?0:offset),focus[1]+(scout?32:narrow?3:2.1),focus[2]+Math.cos(bearing)*back+Math.sin(bearing)*(scout?0:offset));
    camera.position.y=Math.max(camera.position.y,heightAt(world,camera.position.x,camera.position.z)+.75);
    const ahead=scout||narrow?0:3,lateral=!scout&&narrow?-.4:0;
    controls.current?.target.set(focus[0]+Math.sin(bearing)*ahead+Math.cos(bearing)*lateral,focus[1]+(scout?0:.2),focus[2]-Math.cos(bearing)*ahead+Math.sin(bearing)*lateral);lastKey.current=key;
   }
   if(!anim.done&&anim.time>.75){
    const desired=new Vector3(visual[0]-Math.sin(bearing)*9,Math.max(visual[1]+3,heightAt(world,visual[0],visual[2])+3),visual[2]+Math.cos(bearing)*9);
    camera.position.lerp(desired,1-Math.exp(-dt*1.6));controls.current?.target.lerp(new Vector3(...visual),1-Math.exp(-dt*5));
   }
  }
  const center:Vec3=mode==='golf'?(anim.done?ball:anim.start):[t.position.x,t.position.y,t.position.z];
  if(distance(center,anchor)>28)setAnchor([...center]);
  if(sun.current){const center=mode==='golf'?new Vector3(...ball):t.position;sun.current.position.set(center.x-45,center.y+65,center.z-35);sun.current.target.position.copy(center);sun.current.target.updateMatrixWorld();}
  lastMode.current=mode;
 });
 const line=useMemo(()=>{const pts:Vec3[]=[];for(let i=0;i<=16;i++){const x=ball[0]+Math.sin(bearing)*i*.65,z=ball[2]-Math.cos(bearing)*i*.65;pts.push([x,heightAt(world,x,z)+.05,z]);}return pts;},[ball,bearing,world]);
 return <>
  <color attach="background" args={['#a8c5d7']}/><fog attach="fog" args={['#c2d2d5',380,1700]}/>
  <Sky distance={450000} sunPosition={[-180,250,-120]} turbidity={3.2} rayleigh={1.8} mieCoefficient={.003}/>
  <hemisphereLight args={['#e1edfa','#74805a',1.8]}/>
  <directionalLight key={quality} ref={sun} position={[ball[0]-45,ball[1]+65,ball[2]-35]} intensity={3.2} color="#fff1d5" castShadow shadow-mapSize={[quality==='high'?2048:1024,quality==='high'?2048:1024]} shadow-camera-left={quality==='high'?-35:-22} shadow-camera-right={quality==='high'?35:22} shadow-camera-top={quality==='high'?35:22} shadow-camera-bottom={quality==='high'?-35:-22} shadow-camera-near={1} shadow-camera-far={180} shadow-bias={-.00015} shadow-normalBias={.025}/>
  <Terrain world={world}/><Trees course={course} world={world} anchor={anchor} quality={quality}/><Grass world={world} anchor={anchor} quality={quality}/>
  <Golfer actor={golfer} pose={pose}/><Cart actor={cart}/>
  {mode!=='golf'&&<group position={ball}><mesh rotation={[-Math.PI/2,0,0]} position={[0,.07,0]}><ringGeometry args={[1.2,1.4,40]}/><meshBasicMaterial color="#f6d79b" side={2}/></mesh><mesh position={[0,4,0]}><cylinderGeometry args={[.12,.12,8,8]}/><meshBasicMaterial color="#f6d79b" transparent opacity={.22} depthWrite={false}/></mesh></group>}
  <mesh ref={ballMesh} castShadow><sphereGeometry args={[.035,20,14]}/><meshStandardMaterial color="#ffffff" roughness={.4}/></mesh>
  <group position={world.pin}><mesh position={[0,1.35,0]} castShadow><cylinderGeometry args={[.014,.014,2.7,8]}/><meshStandardMaterial color="#e9e8d8"/></mesh><mesh position={[.24,2.45,0]} castShadow><boxGeometry args={[.48,.32,.008]}/><meshStandardMaterial color="#ce4e36" side={2}/></mesh><mesh rotation={[-Math.PI/2,0,0]} position={[0,.012,0]}><circleGeometry args={[.054,24]}/><meshBasicMaterial color="#122114"/></mesh></group>
  {mode==='golf'&&<Line points={line} color="#d6efae" lineWidth={1} dashed dashSize={.15} gapSize={.3}/>}
  {cameraMode==='scout'&&<mesh rotation={[-Math.PI/2,0,0]} position={[preview[0],heightAt(world,preview[0],preview[2])+.15,preview[2]]}><ringGeometry args={[1.8,2.05,48]}/><meshBasicMaterial color="#f5e3a0" side={2}/></mesh>}
  {showGrid&&distance(ball,world.pin)<40&&<GreenGrid world={world} ball={ball}/>}
  {result&&<Line points={result.path.filter((_,i)=>i%4===0).map(p=>[p[0],p[1]+.04,p[2]] as Vec3)} color="#f8efd1" transparent opacity={.32} lineWidth={1.2}/>}
  <OrbitControls ref={controls} enabled={mode==='golf'} enablePan={cameraMode==='scout'} minDistance={1.5} maxDistance={220} maxPolarAngle={Math.PI/2-.05}/>
 </>;
}
export default memo(function CourseScene(props:Props){
 const [surfaceKey,setSurfaceKey]=useState(0);
 const runtime=useRef<Runtime>({animation:{motion:-1,time:0,done:true,start:[...props.ball]},traveler:{position:new Vector3(...props.ball),heading:props.bearing}});
 useEffect(()=>{
  // WebKit can retain a transparent compositor surface after an orientation
  // resize. Recreate that surface while keeping the shot and travel state.
  if(!/AppleWebKit/.test(navigator.userAgent)||/Chrome|Chromium|Edg|OPR/.test(navigator.userAgent))return;
  const orientation=matchMedia('(orientation: portrait)'),reset=()=>setSurfaceKey(k=>k+1);
  orientation.addEventListener('change',reset);return()=>orientation.removeEventListener('change',reset);
 },[]);
 return <Canvas key={surfaceKey} shadows dpr={[1,props.quality==='high'?1.75:1.15]} camera={{position:[props.ball[0],props.ball[1]+3,props.ball[2]+5],fov:55,near:.08,far:5000}} gl={{antialias:true,alpha:false,powerPreference:'high-performance',toneMapping:ACESFilmicToneMapping,toneMappingExposure:1}}><GameView {...props} runtimeRef={runtime}/></Canvas>;
});
