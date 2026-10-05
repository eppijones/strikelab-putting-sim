import {useEffect,useRef,useState} from 'react';
import {ArrowDown,ArrowUp,Target,X} from 'lucide-react';
import type {useSwing} from '../course/useSwing';
import type {SwingMode} from '../course/swing';

/** CSS-pixel travel is independent of screen height and where the gesture begins. */
export const SWING_TRAVEL=96;
type Props={swing:ReturnType<typeof useSwing>;mode:SwingMode;disabled:boolean;busy:boolean;target:number;distance:number};
export default function MyhraSwingControl({swing,mode,disabled,busy,target,distance}:Props){
 const pointer=useRef<{id:number;x:number;y:number;touch:boolean;cancelled:boolean}|null>(null);
 const [touch,setTouch]=useState(()=>matchMedia('(pointer:coarse)').matches);
 const [cancelled,setCancelled]=useState(false);
 const {cancel}=swing,v=swing.view,timing=['power','path','tempo'].includes(v.phase);
 useEffect(()=>{if(disabled){pointer.current=null;cancel();}},[disabled,cancel]);
 useEffect(()=>{const reset=()=>{pointer.current=null;setCancelled(false);cancel();};addEventListener('blur',reset);addEventListener('orientationchange',reset);document.addEventListener('visibilitychange',reset);return()=>{removeEventListener('blur',reset);removeEventListener('orientationchange',reset);document.removeEventListener('visibilitychange',reset);};},[cancel]);
 const clear=()=>{pointer.current=null;setCancelled(false);};
 return <section className={`mh-swing ${v.phase!=='ready'?'active':''} ${cancelled?'cancelling':''}`} aria-label="Swing controls">
  <div className="mh-swing-readout" aria-live="polite">{busy?'Ball in play':cancelled?'Release to cancel':v.phase==='ready'?'Your shot':v.feedback}</div>
  <button className="mh-swing-pad" aria-label={mode==='three-click'?'Start timing swing':touch?'Pull back and release to swing':'Pull back and swing through'} disabled={disabled}
   onPointerDown={e=>{
    if(pointer.current||e.button!==0)return;
    if(mode==='three-click'||timing){swing.press();return;}
    const isTouch=e.pointerType==='touch';setTouch(isTouch);setCancelled(false);
    pointer.current={id:e.pointerId,x:e.clientX,y:e.clientY,touch:isTouch,cancelled:false};
    e.currentTarget.setPointerCapture(e.pointerId);swing.begin(isTouch?'touch':'mouse');
   }}
   onPointerMove={e=>{
    const p=pointer.current;if(!p||p.id!==e.pointerId)return;
    const dx=e.clientX-p.x,dy=e.clientY-p.y;
    p.cancelled=Math.abs(dx)>130||(p.touch&&dy< -36);setCancelled(p.cancelled);
    if(!p.cancelled)swing.move(dy/SWING_TRAVEL,dx/SWING_TRAVEL);
   }}
   onPointerUp={e=>{const p=pointer.current;if(!p||p.id!==e.pointerId)return;clear();if(p.cancelled)swing.cancel();else swing.release();if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);}}
   onPointerCancel={e=>{if(pointer.current?.id===e.pointerId){clear();swing.cancel();}}}
   onLostPointerCapture={e=>{if(pointer.current?.id===e.pointerId){clear();swing.cancel();}}}
   onKeyDown={e=>{if(e.code==='Space'||e.code==='Enter'){e.preventDefault();e.stopPropagation();if(!e.repeat)swing.press();}if(e.code==='Escape'){clear();swing.cancel();}}}>
   <svg viewBox="0 0 100 100" aria-hidden="true"><circle className="mh-swing-track" cx="50" cy="50" r="45"/><circle className="mh-swing-progress" cx="50" cy="50" r="45" style={{strokeDasharray:`${Math.min(1,v.amount)*283} 283`}}/><circle className="mh-swing-target" cx={50+45*Math.cos(target*2*Math.PI)} cy={50+45*Math.sin(target*2*Math.PI)} r="2.4"/></svg>
   <span>{cancelled?<X size={28}/>:v.phase==='ready'?<><ArrowDown size={23}/>{!touch&&<ArrowUp size={23}/>}</>:timing&&v.phase!=='power'?<Target size={26}/>:<b>{Math.round(v.amount*100)}<small>%</small></b>}</span>
  </button>
  {timing&&v.phase!=='power'?<div className="mh-timing-line"><i/><b style={{left:`${50+v.needle*47}%`}}/></div>:<div className="mh-swing-label">{mode==='three-click'?'Start · power · accuracy':touch?'Pull back · release':'Back · through'}</div>}
  <small className="mh-swing-hint">{v.phase==='ready'?`Target ${Math.round(target*100)}% · ${distance<10?distance.toFixed(1):Math.round(distance)} m`:cancelled?'Shot will not count':touch?'Slide sideways to cancel':'Cross the start to strike'}</small>
  <div className="mh-swing-accessible" role="meter" aria-label="Swing power" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(Math.min(1,v.amount)*100)}/>
 </section>;
}
