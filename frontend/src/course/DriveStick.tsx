import {useEffect,useRef,useState,type PointerEvent} from 'react';
import './navigation.css';

export interface DriveAxes {forward:number;turn:number}
export function DriveStick({onMove,disabled=false}:{onMove:(axes:DriveAxes)=>void;disabled?:boolean}){
 const [point,setPoint]=useState({x:0,y:0}),pointer=useRef<number|null>(null),callback=useRef(onMove);useEffect(()=>{callback.current=onMove;},[onMove]);
 const release=()=>{pointer.current=null;setPoint({x:0,y:0});callback.current({forward:0,turn:0});};
 useEffect(()=>{const clear=()=>{pointer.current=null;setPoint({x:0,y:0});callback.current({forward:0,turn:0});};addEventListener('blur',clear);addEventListener('visibilitychange',clear);return()=>{removeEventListener('blur',clear);removeEventListener('visibilitychange',clear);callback.current({forward:0,turn:0});};},[]);
 const move=(e:PointerEvent<HTMLDivElement>)=>{if(disabled||pointer.current!==e.pointerId)return;const r=e.currentTarget.getBoundingClientRect(),radius=r.width*.32,x=(e.clientX-r.x-r.width/2)/radius,y=(e.clientY-r.y-r.height/2)/radius,length=Math.hypot(x,y),scale=Math.max(1,length),px=x/scale,py=y/scale;setPoint({x:px,y:py});const gain=length<.12?0:(Math.min(1,length)-.12)/.88/Math.max(.001,Math.min(1,length));callback.current({forward:-py*gain,turn:px*gain});};
 return <div className="drive-stick-wrap"><div className="drive-stick" role="group" aria-label="Drive joystick" aria-disabled={disabled} onPointerDown={e=>{if(disabled||pointer.current!==null)return;e.preventDefault();pointer.current=e.pointerId;e.currentTarget.setPointerCapture(e.pointerId);move(e);}} onPointerMove={move} onPointerUp={release} onPointerCancel={release} onLostPointerCapture={release}><span className="drive-stick-guide" aria-hidden="true">↑</span><span className="drive-stick-knob" style={{transform:`translate(${point.x*42}px,${point.y*42}px)`}}/></div><span>DRIVE / STEER</span></div>;
}
