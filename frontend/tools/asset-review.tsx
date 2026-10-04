import React,{Suspense,useRef} from 'react';
import {createRoot} from 'react-dom/client';
import {Canvas,useFrame} from '@react-three/fiber';
import {Environment,OrbitControls,Grid} from '@react-three/drei';
import {Group} from 'three';
import {stanceDistance} from '../src/myhra/golfMotion';
import MyhraGolfer from '../src/myhra/MyhraGolfer';
import {GolfCart} from '../src/myhra/MyhraFeatures';
import {SwingController} from '../src/course/swing';
const q=new URLSearchParams(location.search),asset=q.get('asset')||'golfer',phase=q.get('phase')||'ready',angle=q.get('angle')||'front';
function Model(){const actor=useRef<Group>(null),swing=useRef(new SwingController()),time=useRef(-1);swing.current.phase=phase as any;swing.current.amount=Number(q.get('amount')??1);swing.current.lockedPower=1;time.current=Number(q.get('time')??-1);
 useFrame(({scene,camera})=>{(window as any).assetScene=scene;(window as any).assetCamera=camera;});
 return <>{asset==='cart'?<GolfCart actor={actor}/>:<><MyhraGolfer swing={swing} elapsed={time} club={Number(q.get('club')??0)}/><mesh position={[0,.024,stanceDistance(Number(q.get('club')??0))]}><sphereGeometry args={[.02135,24,16]}/><meshStandardMaterial color="white"/></mesh></>}</>;
}
createRoot(document.getElementById('root')!).render(<><label>{asset} / {phase} / {angle}</label><Canvas shadows camera={{position:angle==='side'?[4,1.2,0]:angle==='back'?[0,1.3,-4]:[2.8,1.6,3.5],fov:35}}><color attach="background" args={['#aebcb6']}/><ambientLight intensity={.8}/><directionalLight position={[3,5,5]} intensity={2.8} castShadow/><Suspense fallback={null}><Environment files="/courses/grenland/myhra/morning.hdr"/><Model/></Suspense><mesh rotation={[-Math.PI/2,0,0]} receiveShadow><planeGeometry args={[200,200]}/><meshStandardMaterial color="#8b9a81"/></mesh><OrbitControls target={[0,1,0]}/></Canvas></>);
