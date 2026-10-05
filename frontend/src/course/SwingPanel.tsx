import {useRef} from 'react';
import type {useSwing} from './useSwing';
import type {SwingMode} from './swing';
export default function SwingPanel({swing,mode,disabled,busy}:{swing:ReturnType<typeof useSwing>;mode:SwingMode;disabled:boolean;busy:boolean}){
 const start=useRef<{x:number;y:number;range:number;id:number}|null>(null),v=swing.view;
 const timing=['power','path','tempo'].includes(v.phase);
 return <section className={`swing-panel ${disabled?'unavailable':''}`} aria-label="Swing controls">
   <div className="swing-top"><span>{timing?v.phase.toUpperCase():mode==='analog'?'SWING STICK':'THREE CLICK'}</span><b>{Math.round(v.amount*100)}%</b></div>
   <div className="swing-meter" role="meter" aria-label="Swing power" aria-valuemin={0} aria-valuemax={108} aria-valuenow={Math.round(v.amount*100)}><i style={{width:`${Math.min(100,v.amount*100)}%`}}/><em/></div>
   {(v.phase==='path'||v.phase==='tempo')&&<div className="swing-accuracy"><i/><b style={{left:`${50+v.needle*47}%`}}/></div>}
   <button className="swing-pad" disabled={disabled} aria-label={mode==='analog'?'Drag down then up to swing':'Click swing timing'}
    onPointerDown={e=>{if(e.button!==0||start.current)return;if(mode==='analog'&&!timing){e.currentTarget.setPointerCapture(e.pointerId);start.current={x:e.clientX,y:e.clientY,range:80,id:e.pointerId};swing.begin(e.pointerType==='touch'?'touch':'mouse');}else swing.press();}}
    onPointerMove={e=>{if(start.current?.id===e.pointerId)swing.move((e.clientY-start.current.y)/start.current.range,(e.clientX-start.current.x)/100);}}
    onPointerUp={()=>{if(start.current){swing.release();start.current=null;}}}
    onPointerCancel={()=>{start.current=null;swing.reset();}}
    onLostPointerCapture={()=>{if(start.current){start.current=null;swing.cancel();}}}
    onKeyDown={e=>{if(e.code==='Space'||e.code==='Enter'){e.preventDefault();if(!e.repeat)swing.press();}}}>
    <span>{disabled?(busy?'Ball in flight…':'Preparing course…'):v.phase==='ready'?(mode==='analog'?'↓ BACK · ↑ THROUGH':'START SWING'):v.feedback}</span>
    <small>{mode==='analog'?'Touch: pull / release · mouse / stick: back / through':'Space / tap / ✕ · start, power, accuracy'}</small>
   </button>
 </section>;
}
