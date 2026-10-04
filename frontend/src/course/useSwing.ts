import {useCallback,useEffect,useRef,useState} from 'react';
import {SwingController,type Strike,type SwingMode} from './swing';
export function useSwing(onStrike:(strike:Strike)=>void,enabled:boolean,mode:SwingMode){
  const controller=useRef(new SwingController());
  const callback=useRef(onStrike),allowed=useRef(enabled);
  useEffect(()=>{callback.current=onStrike;allowed.current=enabled;if(!enabled&&controller.current.phase!=='finish')controller.current.reset();},[onStrike,enabled]);
  const [view,setView]=useState(()=>new SwingController().view());
  useEffect(()=>{controller.current.reset(mode);setView(controller.current.view());},[mode]);
  useEffect(()=>{let raf=0,last=0;const frame=(now:number)=>{controller.current.update(now);if(now-last>33){const next=controller.current.view();setView(previous=>JSON.stringify(previous)===JSON.stringify(next)?previous:next);last=now;}raf=requestAnimationFrame(frame);};raf=requestAnimationFrame(frame);return()=>cancelAnimationFrame(raf);},[]);
  const emit=useCallback((strike:Strike|undefined)=>{if(strike)callback.current(strike);},[]);
  const begin=useCallback(()=>{if(allowed.current)controller.current.begin(performance.now());},[]);
  const move=useCallback((pull:number,side:number)=>{if(allowed.current)emit(controller.current.move(pull,side,performance.now()));},[emit]);
  const release=useCallback(()=>{if(allowed.current)emit(controller.current.release(performance.now()));},[emit]);
  const press=useCallback(()=>{if(allowed.current)emit(controller.current.press(performance.now()));},[emit]);
  const reset=useCallback(()=>controller.current.reset(),[]);
  return {view,controller,begin,move,release,press,reset};
}
