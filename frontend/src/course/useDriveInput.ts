import {useCallback,useEffect,useRef,useState} from 'react';
import type {DriveAxes} from './DriveStick';

export interface TravelInput extends DriveAxes {boost:boolean}
/** Merge input sources so a connected, idle controller cannot erase keyboard/touch. */
export function useDriveInput(active:boolean){
 const input=useRef<TravelInput>({forward:0,turn:0,boost:false}),touch=useRef<DriveAxes>({forward:0,turn:0}),[boost,setBoost]=useState(false),boostRef=useRef(false);useEffect(()=>{boostRef.current=boost;},[boost]);
 const onTouch=useCallback((value:DriveAxes)=>{touch.current=value;},[]);
 useEffect(()=>{const keys=new Set<string>();let frame=0,padId='',padArmed=false;const reset=()=>{keys.clear();touch.current={forward:0,turn:0};input.current={forward:0,turn:0,boost:false};padArmed=false;};
  const down=(e:KeyboardEvent)=>{if(!active||(e.target as HTMLElement)?.closest('input,select,textarea'))return;if(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','ShiftLeft','ShiftRight'].includes(e.code)){e.preventDefault();keys.add(e.code);}};
  const up=(e:KeyboardEvent)=>keys.delete(e.code),dead=(n:number)=>Math.abs(n)<.16?0:n,clamp=(n:number)=>Math.max(-1,Math.min(1,n));
  const tick=()=>{const connected=Array.from(navigator.getGamepads?.()??[]).find(p=>p?.connected),id=connected?`${connected.index}:${connected.id}`:'';if(id!==padId){padId=id;padArmed=false;}if(connected&&!padArmed&&Math.abs(connected.axes[0]??0)<.16&&Math.abs(connected.axes[1]??0)<.16&&!connected.buttons[7]?.pressed)padArmed=true;const pad=padArmed?connected:undefined;input.current={forward:active?clamp(touch.current.forward+Number(keys.has('KeyW')||keys.has('ArrowUp'))-Number(keys.has('KeyS')||keys.has('ArrowDown'))-dead(pad?.axes[1]??0)):0,turn:active?clamp(touch.current.turn+Number(keys.has('KeyD')||keys.has('ArrowRight'))-Number(keys.has('KeyA')||keys.has('ArrowLeft'))+dead(pad?.axes[0]??0)):0,boost:active&&(boostRef.current||keys.has('ShiftLeft')||keys.has('ShiftRight')||!!pad?.buttons[7]?.pressed)};frame=requestAnimationFrame(tick);};
  addEventListener('keydown',down);addEventListener('keyup',up);addEventListener('blur',reset);addEventListener('orientationchange',reset);addEventListener('visibilitychange',reset);tick();return()=>{cancelAnimationFrame(frame);removeEventListener('keydown',down);removeEventListener('keyup',up);removeEventListener('blur',reset);removeEventListener('orientationchange',reset);removeEventListener('visibilitychange',reset);reset();};
 },[active]);
 return {input,onTouch,boost,setBoost};
}
