import {useEffect,useRef} from 'react';
import {useThree} from '@react-three/fiber';

/** Input changes offsets only. CourseView is the sole camera transform owner. */
export function useCameraInput(enabled:boolean,reset:number){
 const {gl}=useThree(),orbit=useRef({yaw:0,pitch:0,zoom:1,reset,gesture:'none' as 'none'|'aim'|'orbit'});
 useEffect(()=>{orbit.current={yaw:0,pitch:0,zoom:1,reset,gesture:'none'};},[reset]);
 useEffect(()=>{
  const canvas=gl.domElement,points=new Map<number,{x:number;y:number;button:number}>();
  const down=(e:PointerEvent)=>{points.set(e.pointerId,{x:e.clientX,y:e.clientY,button:e.button});if(e.button===2||points.size>1)orbit.current.gesture='orbit';else if(points.size===1)orbit.current.gesture='aim';};
  const move=(e:PointerEvent)=>{const p=points.get(e.pointerId);if(!p)return;
   if(enabled&&orbit.current.gesture==='orbit'){
    const others=[...points.entries()].filter(([id])=>id!==e.pointerId).map(([,p])=>p);
    if(others.length){const other=others[0],before=Math.hypot(p.x-other.x,p.y-other.y),after=Math.hypot(e.clientX-other.x,e.clientY-other.y);if(before>12&&after>12)orbit.current.zoom=Math.max(.65,Math.min(1.8,orbit.current.zoom*before/after));}
    orbit.current.yaw-=(e.clientX-p.x)*.004;orbit.current.pitch=Math.max(-.35,Math.min(.85,orbit.current.pitch+(e.clientY-p.y)*.003));
   }
   p.x=e.clientX;p.y=e.clientY;
  };
  const up=(e:PointerEvent)=>{points.delete(e.pointerId);if(!points.size)orbit.current.gesture='none';},clear=()=>{points.clear();orbit.current.gesture='none';},menu=(e:Event)=>e.preventDefault();
  const wheel=(e:WheelEvent)=>{if(!enabled)return;e.preventDefault();orbit.current.zoom=Math.max(.65,Math.min(1.8,orbit.current.zoom*Math.exp(e.deltaY*.001)));};
  canvas.addEventListener('pointerdown',down);canvas.addEventListener('wheel',wheel,{passive:false});canvas.addEventListener('contextmenu',menu);window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);window.addEventListener('pointercancel',up);window.addEventListener('blur',clear);window.addEventListener('orientationchange',clear);document.addEventListener('visibilitychange',clear);
  return()=>{canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('wheel',wheel);canvas.removeEventListener('contextmenu',menu);window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);window.removeEventListener('pointercancel',up);window.removeEventListener('blur',clear);window.removeEventListener('orientationchange',clear);document.removeEventListener('visibilitychange',clear);};
 },[gl,enabled]);
 return orbit;
}
