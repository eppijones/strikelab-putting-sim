/* eslint-disable react-refresh/only-export-components -- standalone art-review entry */
import React,{StrictMode,Suspense,useEffect,useRef} from 'react';
import {createRoot} from 'react-dom/client';
import {Canvas,useFrame} from '@react-three/fiber';
import {Environment,OrbitControls,useGLTF} from '@react-three/drei';
import {Group,type Scene,type Camera} from 'three';
import {stanceDistance} from '../src/myhra/golfMotion';
import MyhraGolfer from '../src/myhra/MyhraGolfer';
import {GolfCart} from '../src/myhra/MyhraFeatures';
import {SwingController} from '../src/course/swing';
import {terrainGeometry} from '../src/course/terrain';
import type {World} from '../src/course/engine';
declare global {interface Window {assetScene?:Scene;assetCamera?:Camera;reviewTime?:React.MutableRefObject<number>;reviewSwing?:React.MutableRefObject<SwingController>}}
const q=new URLSearchParams(location.search),asset=q.get('asset')||'golfer',phase=q.get('phase')||'ready',angle=q.get('angle')||'front';
const slope=Number(q.get('slope')??0),stance=stanceDistance(Number(q.get('club')??0));
const reviewWorld:World|undefined=q.has('slope')?{terrain:{url:'authored QA plane',surfaces:'authored QA plane',size:3,spacing:5,origin:[-5,-5],heights:Float32Array.from({length:9},(_,i)=>slope*(Math.floor(i/3)*5-5-stance)),lies:new Uint8Array(9)},pin:[0,0,stance],stimp:10,wind:[0,0]}:undefined;
const reviewGround=reviewWorld?terrainGeometry(reviewWorld.terrain,undefined,10):undefined;
function Fir(){const {scene}=useGLTF('/courses/grenland/myhra/fir-near.glb');return <primitive object={scene}/>;}
function Model(){const actor=useRef<Group>(null),swing=useRef(new SwingController()),time=useRef(-1);
 useEffect(()=>{const allowed=['ready','backswing','downswing','power','path','tempo','finish'];if(allowed.includes(phase))swing.current.phase=phase as SwingController['phase'];swing.current.amount=Number(q.get('amount')??1);swing.current.peak=swing.current.amount;swing.current.lockedPower=1;time.current=Number(q.get('time')??-1);window.reviewTime=time;window.reviewSwing=swing;return()=>{delete window.reviewTime;delete window.reviewSwing;};},[]);
 useFrame(({scene,camera})=>{window.assetScene=scene;window.assetCamera=camera;});
 return <>{asset==='cart'?<GolfCart actor={actor}/>:asset==='fir'?<Fir/>:<><MyhraGolfer swing={swing} elapsed={time} club={Number(q.get('club')??0)} world={reviewWorld} high={q.get('quality')!=='mobile'} shotType={q.get('style')==='chip'?'chip':'full'}/><mesh position={[0,.024,stanceDistance(Number(q.get('club')??0))]}><sphereGeometry args={[.02135,24,16]}/><meshStandardMaterial color="white"/></mesh></>}</>;
}
createRoot(document.getElementById('root')!).render(<StrictMode><label>{asset} / {phase} / {angle}</label><Canvas shadows camera={{position:asset==='fir'?[18,10,25]:q.has('close')?angle==='side'?[1.3,1.1,.2]:angle==='back'?[0,1.15,-1.1]:[.7,1.15,1.6]:angle==='side'?[4,1.2,0]:angle==='back'?[0,1.3,-4]:[2.8,1.6,3.5],fov:35}}><color attach="background" args={['#aebcb6']}/><ambientLight intensity={.8}/><directionalLight position={[3,5,5]} intensity={2.8} castShadow shadow-bias={-.0002} shadow-normalBias={.015}/><Suspense fallback={null}><Environment files="/courses/grenland/myhra/morning.hdr"/><Model/></Suspense>{reviewGround?<mesh geometry={reviewGround} receiveShadow><meshStandardMaterial color="#8b9a81"/></mesh>:<mesh rotation={[-Math.PI/2,0,0]} receiveShadow><planeGeometry args={[200,200]}/><meshStandardMaterial color="#8b9a81"/></mesh>}<OrbitControls target={asset==='fir'?[0,7,0]:q.has('close')?[0,1.02,.32]:[0,1,0]}/></Canvas></StrictMode>);
