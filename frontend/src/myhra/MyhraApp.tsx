import {lazy,Suspense,useCallback,useEffect,useMemo,useRef,useState} from 'react';
import {ArrowDown,ArrowRight,ArrowUp,CarFront,Check,ChevronDown,ChevronLeft,ChevronRight,Flag,HelpCircle,Menu,RotateCcw,Sun,Target,Volume2,VolumeX,Wind,X} from 'lucide-react';
import {CLUBS,bearingTo,distance,heightAt,scoreName,simulate,surfaceAt,type ShotResult,type Vec3} from '../course/engine';
import {useSwing} from '../course/useSwing';
import type {Strike,SwingMode} from '../course/swing';
import {IMPACT_DELAY} from './golfMotion';
import {golfSound} from '../course/audio';
import {SAVE,freshHole,loadHole,restoreHole,type Hole,type SavedHole} from './data';
import {ErrorBoundary} from '../components/shared/ErrorBoundary';
import type {Advice} from './caddie';
import {DriveStick} from '../course/DriveStick';
import {useDriveInput} from '../course/useDriveInput';
import CourseMap from '../course/CourseMap';
import './myhra.css';

const Scene=lazy(()=>import('./MyhraScene'));
type View='player'|'overview'|'green';
const roundValue=(strokes:number)=>strokes===4?'E':`${strokes>4?'+':''}${strokes-4}`;
const clamp=(n:number,min:number,max:number)=>Math.max(min,Math.min(max,n));

function MiniMap({hole,ball}:{hole:Hole;ball:Vec3}){
 const point=(p:number[])=>`${(p[0]-290)*.68},${(p[1]-195)*.43}`;
 const regions=hole.data.regions.filter(r=>r.points.some(p=>p[0]>330&&p[0]<460&&p[1]>220&&p[1]<590));
 return <svg viewBox="0 0 136 178" aria-label="Hole map" role="img"><defs><linearGradient id="mapFade" x2="0" y2="1"><stop stopColor="#152e2d"/><stop offset="1" stopColor="#0d201e"/></linearGradient></defs><rect width="136" height="178" rx="5" fill="url(#mapFade)"/>{regions.map(r=><polygon key={r.id} points={r.points.map(point).join(' ')} fill={r.kind==='water'?'#74a3ad':r.kind==='green'?'#a7b991':r.kind==='bunker'?'#e1d7b7':'#4b6b4d'} opacity={r.kind==='bunker'?.9:.7}/>)}<path d={`M ${point([ball[0],ball[2]])} L ${point([hole.data.pin[0],hole.data.pin[2]])}`} fill="none" stroke="#e6dbb8" opacity=".3" strokeDasharray="2 3"/><circle cx={(hole.data.pin[0]-290)*.68} cy={(hole.data.pin[2]-195)*.43} r="3" fill="#e6c577"/><circle cx={(ball[0]-290)*.68} cy={(ball[2]-195)*.43} r="4" fill="#fff" stroke="#20352b" strokeWidth="1.5"/></svg>;
}

function SwingControl({swing,mode,disabled,busy}:{swing:ReturnType<typeof useSwing>;mode:SwingMode;disabled:boolean;busy:boolean}){
 const pointer=useRef<{x:number;y:number;range:number}|null>(null),v=swing.view;
 const timing=['power','path','tempo'].includes(v.phase);
 return <div className={`mh-swing ${v.phase!=='ready'?'active':''}`}>
  <div className="mh-swing-readout" aria-live="polite">{busy?'WATCH YOUR SHOT':v.phase==='ready'?'READY WHEN YOU ARE':v.feedback.toUpperCase()}</div>
  <button className="mh-swing-pad" aria-label={mode==='analog'?'Pull back and swing through':'Start timing swing'} disabled={disabled}
   onPointerDown={e=>{if(mode==='analog'&&!timing){e.currentTarget.setPointerCapture(e.pointerId);pointer.current={x:e.clientX,y:e.clientY,range:Math.max(30,Math.min(110,innerHeight-e.clientY-10))};swing.begin();}else swing.press();}}
   onPointerMove={e=>{if(pointer.current)swing.move((e.clientY-pointer.current.y)/pointer.current.range,(e.clientX-pointer.current.x)/110);}}
   onPointerUp={()=>{if(pointer.current){swing.release();pointer.current=null;}}} onPointerCancel={()=>{pointer.current=null;swing.reset();}}
   onKeyDown={e=>{if(e.code==='Space'||e.code==='Enter'){e.preventDefault();e.stopPropagation();if(!e.repeat)swing.press();}}}>
   <svg viewBox="0 0 100 100" aria-hidden="true"><circle className="mh-swing-track" cx="50" cy="50" r="45"/><circle className="mh-swing-progress" cx="50" cy="50" r="45" style={{strokeDasharray:`${Math.min(1,v.amount)*283} 283`}}/></svg>
   <span>{v.phase==='ready'?<><ArrowDown size={18}/><ArrowUp size={18}/></>:timing&&v.phase!=='power'?<Target size={24}/>:<b>{Math.round(v.amount*100)}<small>%</small></b>}</span>
  </button>
  {timing&&v.phase!=='power'?<div className="mh-timing-line"><i/><b style={{left:`${50+v.needle*47}%`}}/></div>:<div className="mh-swing-label">{mode==='analog'?'PULL BACK · SWING THROUGH':'TAP · POWER · LINE · TEMPO'}</div>}
  <div className="mh-swing-accessible" role="meter" aria-label="Swing power" aria-valuemin={0} aria-valuemax={108} aria-valuenow={Math.round(v.amount*100)}/>
 </div>;
}

function PlayHole({hole}:{hole:Hole}){
 const [round,setRound]=useState<SavedHole>(()=>{try{return restoreHole(hole,localStorage.getItem(SAVE));}catch{return freshHole(hole);}});
 const [club,setClub]=useState(()=>surfaceAt(hole.world,round.ball[0],round.ball[2])==='green'?13:0);
 const [power,setPower]=useState(1),[aim,setAim]=useState(0),[view,setView]=useState<View>('player');
 const [shot,setShot]=useState<ShotResult>(),[motion,setMotion]=useState(0),[busy,setBusy]=useState(false),[ready,setReady]=useState(false);
 const [map,setMap]=useState(false),[cameraReset,setCameraReset]=useState(0),[menu,setMenu]=useState(false),[help,setHelp]=useState(false),[bag,setBag]=useState(false),[card,setCard]=useState(false);
 const [sound,setSound]=useState(true),[quality,setQuality]=useState<'balanced'|'high'>(()=>matchMedia('(pointer:coarse)').matches?'balanced':'high');
 const [swingMode,setSwingMode]=useState<SwingMode>('analog'),[feedback,setFeedback]=useState(''),[gamepad,setGamepad]=useState(false),[saveError,setSaveError]=useState(false);
 const [guidance,setGuidance]=useState<{key:string;advice?:Advice;error?:boolean}>(),[caddieOpen,setCaddieOpen]=useState(false);
 const [cartSpeed,setCartSpeed]=useState(0),[cartDistance,setCartDistance]=useState(0),[hour,setHour]=useState(10),[dayCycle,setDayCycle]=useState(false),[cartMode,setCartMode]=useState(false),[friendly,setFriendly]=useState(true);
 const {input:travelRef,boost:driveBoost,setBoost:setDriveBoost,onTouch:onDriveTouch}=useDriveInput(cartMode&&!menu&&!help&&!card&&!bag&&!map),dialog=useRef<HTMLElement>(null);
 const travelStatus=useCallback((speed:number,metres:number)=>{setCartSpeed(speed);setCartDistance(metres);},[]);
 const closeMap=useCallback(()=>setMap(false),[]);
 const ball=round.ball,pin=hole.data.pin,remaining=distance(ball,pin),lie=surfaceAt(hole.world,ball[0],ball[2]);
 const aimTarget:Vec3=useMemo(()=>round.strokes===0&&distance(ball,hole.data.tee)<2?[355,heightAt(hole.world,355,370),370]:pin,[round.strokes,ball,hole,pin]);
 const adviceKey=round.ball.join(','),advice=guidance?.key===adviceKey?guidance.advice:undefined,caddieError=guidance?.key===adviceKey&&guidance.error;
 const bearing=bearingTo(ball,aimTarget)+aim*Math.PI/180;
 const clubSpeed=CLUBS[club].speed*(club===13?clamp(Math.sqrt(remaining/7),.25,1):1);
 const selectedSpeed=clubSpeed*power*(lie==='rough'&&club!==13?.84:lie==='bunker'&&club!==13?.6:1);
 const preview=useMemo(()=>simulate(hole.world,ball,{speed:selectedSpeed,bearing,launch:CLUBS[club].loft,spin:CLUBS[club].spin}),[hole,ball,selectedSpeed,bearing,club]);
 const allowed=ready&&!busy&&!cartMode&&!round.complete&&!menu&&!help&&!card&&!bag&&!map;
 useEffect(()=>{
  if(round.complete)return;
  const worker=new Worker(new URL('./caddie.worker.ts',import.meta.url),{type:'module'});
  worker.onmessage=e=>setGuidance({key:adviceKey,advice:e.data.advice,error:!!e.data.error});worker.onerror=()=>setGuidance({key:adviceKey,error:true});
  worker.postMessage({id:round.strokes,world:hole.world,ball:round.ball});return()=>worker.terminate();
 },[hole,round.ball,round.strokes,round.complete,adviceKey]);
 useEffect(()=>{if(!dayCycle)return;const timer=setInterval(()=>setHour(h=>h>=19.5?6.5:h+1/60),1000);return()=>clearInterval(timer);},[dayCycle]);
 const commit=useCallback((next:SavedHole)=>{setRound(next);try{localStorage.setItem(SAVE,JSON.stringify(next));setSaveError(false);}catch{setSaveError(true);}},[]);
 const onStrike=useCallback((strike:Strike)=>{
  if(!allowed)return;
  const amount=friendly&&strike.power>.9?1:strike.power,contact=friendly?1-(1-strike.contact)*.25:strike.contact;
  const result=simulate(hole.world,ball,{speed:selectedSpeed*amount*contact,bearing:bearing+strike.face*(friendly?.3:1)*Math.PI/180,launch:CLUBS[club].loft,spin:CLUBS[club].spin});
  setShot(result);setMotion(n=>n+1);setBusy(true);setView('player');setFeedback(strike.label);if(sound)setTimeout(()=>golfSound('hit'),IMPACT_DELAY*1000);
  commit({...round,ball:result.end,strokes:round.strokes+1+result.penalty,complete:result.made,history:[...round.history,{club:CLUBS[club].name,distance:result.distance,carry:result.carry,lie,penalty:result.penalty}]});
 },[allowed,hole,ball,selectedSpeed,bearing,club,sound,commit,round,lie,friendly]);
 const swing=useSwing(onStrike,allowed,swingMode);
 const {reset:swingReset,press:swingPress,begin:swingBegin,move:swingMove,controller:swingController}=swing;
 const settled=useCallback(()=>{
  setBusy(false);swingReset();setAim(0);
  if(round.complete){if(sound)golfSound('cup');return;}
  const d=distance(round.ball,pin),l=surfaceAt(hole.world,round.ball[0],round.ball[2]);
  const next=l==='green'?13:d<55?11:d<100?9:d<135?7:d<170?5:d<205?2:0;
  setClub(next);setPower(1);setCaddieOpen(false);
 },[round,pin,hole,sound,swingReset]);
 const sceneReady=useCallback(()=>setReady(true),[]);
 const aimTo=useCallback((point:Vec3)=>{if(!allowed||swingController.current.phase!=='ready')return;const base=bearingTo(ball,aimTarget),a=bearingTo(ball,point)-base;setAim(clamp(Math.atan2(Math.sin(a),Math.cos(a))*180/Math.PI,-180,180));},[allowed,ball,aimTarget,swingController]);
 const restart=()=>{setCaddieOpen(false);setCartMode(false);travelRef.current={forward:0,turn:0,boost:false};swingReset();setShot(undefined);setBusy(false);commit(freshHole(hole));setClub(0);setPower(1);setAim(0);setView('player');setMenu(false);setFeedback('');};
 const practice=(metres:number)=>{setCaddieOpen(false);setCartMode(false);travelRef.current={forward:0,turn:0,boost:false};const p:Vec3=metres===70?[365,0,pin[2]+68]:[pin[0],0,pin[2]+metres];p[1]=heightAt(hole.world,p[0],p[2]);swingReset();setShot(undefined);setBusy(false);commit({...freshHole(hole),ball:p});setClub(metres<10?13:11);setPower(metres===70?.9:1);setAim(0);setMenu(false);setView('player');};
 const caddie=()=>{if(!advice)return;setClub(advice.club);setPower(advice.power);const a=advice.bearing-bearingTo(ball,aimTarget);setAim(Math.atan2(Math.sin(a),Math.cos(a))*180/Math.PI);setMenu(false);setFeedback('CADDIE LINE SET · MAKE A FULL SWING');setCaddieOpen(false);};
 const toggleCart=()=>{travelRef.current={forward:0,turn:0,boost:false};swingReset();setCartMode(v=>!v);setView('player');};
 useEffect(()=>{
  const key=(e:KeyboardEvent)=>{
   if(e.code==='Escape'){setMenu(false);setHelp(false);setBag(false);setCard(false);setMap(false);swingReset();return;}
   if(menu||help||card||map){if(e.code==='Tab'&&dialog.current){const nodes=Array.from(dialog.current.querySelectorAll<HTMLElement>('button:not(:disabled),a,input,select'));const first=nodes[0],last=nodes[nodes.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus();}}return;}
   if((e.target as HTMLElement)?.closest('input,select,textarea,a')||e.code==='Space'&&(e.target as HTMLElement)?.closest('button'))return;
   if(cartMode)return;
   if(!allowed)return;
   if(e.code==='Space'){e.preventDefault();if(!e.repeat)swingPress();return;}
   if(swingController.current.phase!=='ready')return;
   if(swingController.current.phase==='ready'&&(e.code==='ArrowLeft'||e.code==='KeyA')){e.preventDefault();setAim(v=>clamp(v-.5,-180,180));}
   if(swingController.current.phase==='ready'&&(e.code==='ArrowRight'||e.code==='KeyD')){e.preventDefault();setAim(v=>clamp(v+.5,-180,180));}
   if(e.code==='ArrowUp'){e.preventDefault();setPower(v=>clamp(v+.02,.1,1));}
   if(e.code==='ArrowDown'){e.preventDefault();setPower(v=>clamp(v-.02,.1,1));}
   if(e.code==='KeyC'&&!e.repeat)setClub(v=>(v+1)%CLUBS.length);
   if(e.code==='KeyR'&&!e.repeat){setView('player');setCameraReset(v=>v+1);}
   if(e.code==='KeyV'&&!e.repeat)setView(v=>v==='player'?'overview':'player');
  };
  const blur=()=>{swingReset();travelRef.current={forward:0,turn:0,boost:false};};addEventListener('keydown',key);addEventListener('blur',blur);return()=>{removeEventListener('keydown',key);removeEventListener('blur',blur);};
 },[allowed,cartMode,menu,help,card,map,swingPress,swingReset,swingController,travelRef]);
 useEffect(()=>{if(!menu&&!help&&!card)return;const prior=document.activeElement as HTMLElement;travelRef.current={forward:0,turn:0,boost:false};dialog.current?.querySelector<HTMLElement>('button')?.focus();return()=>prior?.focus();},[menu,help,card,travelRef]);
 const buttons=useRef<boolean[]>([]),padSwing=useRef(false),hadPad=useRef(false);
 useEffect(()=>{
  let raf=0,last=0;const frame=(now:number)=>{
   const dt=Math.min((now-last)/1000,.05);last=now;const pad=Array.from(navigator.getGamepads?.()??[]).find(Boolean);
   setGamepad(!!pad);
   if(pad){hadPad.current=true;const fresh=(i:number)=>pad.buttons[i]?.pressed&&!buttons.current[i];
    if(allowed){
     if(Math.abs(pad.axes[0])>.2&&swingController.current.phase==='ready')setAim(v=>clamp(v+pad.axes[0]*dt*18,-180,180));
     if(Math.abs(pad.axes[1])>.2&&swingController.current.phase==='ready')setPower(v=>clamp(v-pad.axes[1]*dt*.35,.1,1));
     if(swingController.current.phase==='ready'){if(fresh(4))setClub(v=>(v+13)%14);if(fresh(5))setClub(v=>(v+1)%14);}
     if(fresh(0))swingPress();if(fresh(2))setView(v=>v==='player'?'overview':'player');
     if(pad.axes[3]>.12&&swingController.current.phase==='ready'){swingBegin();padSwing.current=true;}
     if(padSwing.current){swingMove(Math.max(0,pad.axes[3]),pad.axes[2]);if(swingController.current.phase==='finish')padSwing.current=false;}
    }else if(padSwing.current){padSwing.current=false;swingReset();}
    if(fresh(9))setMenu(v=>!v);if(fresh(1)){setMenu(false);setHelp(false);setBag(false);setCard(false);setMap(false);}
    buttons.current=pad.buttons.map(b=>b.pressed);
   }else if(hadPad.current){hadPad.current=false;padSwing.current=false;swingReset();travelRef.current={forward:0,turn:0,boost:false};buttons.current=[];}
   raf=requestAnimationFrame(frame);
  };raf=requestAnimationFrame(frame);return()=>cancelAnimationFrame(raf);
 },[allowed,cartMode,menu,help,card,swingBegin,swingMove,swingPress,swingReset,swingController,travelRef]);
 const shownBall=busy&&shot?shot.path[0]:ball,shownRemaining=distance(shownBall,pin),shownLie=surfaceAt(hole.world,shownBall[0],shownBall[2]);
 const shownStroke=busy?round.strokes-(shot?.penalty??0):round.complete?round.strokes:round.strokes+1;
 const score=round.complete&&!busy?roundValue(round.strokes):'E';
 return <main className={`myhra-game ${busy?'mh-playing':''} ${cartMode?'mh-cart-mode':''}`}>
  <div className="mh-world"><ErrorBoundary><Suspense fallback={<div className="mh-loading"><span className="mh-emblem">G</span><p>Preparing Myhra</p><small>A moment on the first tee.</small></div>}><Scene hole={hole} ball={ball} bearing={bearing} result={shot} motion={motion} swing={swingController} club={club} view={view} quality={quality} onReady={sceneReady} onSettled={settled} onAim={aimTo} preview={preview.end} busy={busy} hour={hour} cartMode={cartMode} travel={travelRef} drivingAllowed={cartMode&&!menu&&!help&&!card&&!bag&&!map} cameraReset={cameraReset} onTravel={travelStatus}/></Suspense></ErrorBoundary></div>
  <div className="mh-vignette"/>
  <header className="mh-header"><a className="mh-wordmark" href="/play/grenland/myhra"><span className="mh-emblem">G<span>•</span></span><span>GRENLAND<small>GOLF CLUB · NORWAY</small></span></a><div className="mh-hole-title"><span>THE OPENING HOLE</span><h1>Myhra</h1></div><nav aria-label="Game menu"><button aria-label="Full course map" disabled={busy} onClick={()=>setMap(true)}><Flag size={18}/></button><button aria-label="Controls" onClick={()=>setHelp(true)}><HelpCircle size={19}/></button><button aria-label="Menu" onClick={()=>setMenu(true)}><Menu size={22}/></button></nav></header>
  <div className="mh-score-strip"><span className="mh-hole-number">01</span><span>PAR <b>4</b></span><i/><span>{hole.data.distance} <small>m</small></span><button aria-label="Scorecard" onClick={()=>setCard(true)}><span>STROKE <b>{shownStroke}</b></span><strong>{score}</strong></button></div>
  <div className="mh-wind"><Wind size={20}/><span>CALM<small>{String(Math.floor(hour)).padStart(2,"0")}:{String(Math.floor((hour%1)*60)).padStart(2,"0")} · {hour<11?"Morning":hour<16?"Daylight":"Evening"}</small></span></div>
  <aside className="mh-pin-info"><span><Flag size={12}/> TO THE PIN</span><strong>{shownRemaining<10?shownRemaining.toFixed(1):Math.round(shownRemaining)}<small>m</small></strong><p>{shownLie==='rough'?'In the rough':shownLie==='green'?'On the green':shownLie==='bunker'?'In the bunker':distance(shownBall,hole.data.tee)<2?'Teeing ground':'On the fairway'}<i/> {Math.abs(pin[1]-shownBall[1]).toFixed(1)} m {pin[1]>=shownBall[1]?'uphill':'downhill'}</p></aside>
  {!cartMode&&!busy&&<nav className="mh-tools" aria-label="Shot tools"><button aria-label={view==='player'?'Scout the hole':'Return to golfer'} aria-pressed={view!=='player'} onClick={()=>setView(v=>v==='player'?'overview':'player')}><Target size={17}/><span>{view==='player'?'Scout':'Player'}</span></button><button aria-label="Drive the cart" onClick={toggleCart}><CarFront size={17}/><span>Cart</span></button><button aria-label="Caddie" aria-expanded={caddieOpen} disabled={round.complete} onClick={()=>setCaddieOpen(v=>!v)}><HelpCircle size={17}/><span>Caddie</span></button>{remaining<40&&!round.complete&&<button aria-label="Read the green" aria-pressed={view==='green'} onClick={()=>setView(v=>v==='green'?'player':'green')}><Flag size={17}/><span>Green</span></button>}<button aria-label="Reset shot camera" title="Reset view · R" onClick={()=>{setView('player');setCameraReset(v=>v+1);}}><RotateCcw size={17}/></button></nav>}
  <button className="mh-minimap" aria-label="Open full course map" disabled={busy} onClick={()=>setMap(true)}><MiniMap hole={hole} ball={shownBall}/><span>COURSE MAP ↗</span></button>
  {!cartMode&&(!round.complete||busy)?<footer className="mh-shot-hud">
   <section className="mh-club"><span className="mh-eyebrow">{club===13?'ON THE GREEN':'IN YOUR HANDS'}</span><button className="mh-club-button" aria-label="Choose club" disabled={busy||swing.view.phase!=='ready'} onClick={()=>setBag(v=>!v)}><strong>{CLUBS[club].name}</strong><ChevronDown size={20}/></button><span className="mh-carry">{Math.round(preview.distance)} m <small>projected total</small></span></section>
   <section className="mh-shot-adjust"><div><label htmlFor="mh-power">POWER <b>{Math.round(power*100)}%</b></label><input id="mh-power" aria-label="Target power" type="range" min="10" max="100" value={power*100} disabled={busy||swing.view.phase!=='ready'} onChange={e=>setPower(+e.target.value/100)}/></div><div className="mh-aim"><button aria-label="Aim left" disabled={busy||swing.view.phase!=='ready'} onClick={()=>setAim(v=>clamp(v-1,-180,180))}><ChevronLeft size={17}/></button><span>{Math.abs(aim)<.1?'STRAIGHT LINE':`${Math.abs(aim).toFixed(1)}° ${aim<0?'LEFT':'RIGHT'}`}</span><button aria-label="Aim right" disabled={busy||swing.view.phase!=='ready'} onClick={()=>setAim(v=>clamp(v+1,-180,180))}><ChevronRight size={17}/></button></div></section>
   <SwingControl swing={swing} mode={swingMode} disabled={!allowed} busy={busy}/>
  </footer>:!cartMode&&<section className="mh-result" aria-label="Hole completed"><span className="mh-eyebrow">MYHRA · HOLE COMPLETE</span><h2>{scoreName(round.strokes,4)}</h2><p><strong>{round.strokes}</strong> strokes <span>{roundValue(round.strokes)}</span></p><button onClick={restart}><RotateCcw size={16}/> Play Myhra again</button><button className="mh-quiet" onClick={()=>setCard(true)}>See your shots <ArrowRight size={15}/></button></section>}
  {feedback&&!busy&&!round.complete&&<div className="mh-shot-recap"><span>{feedback}</span>{round.strokes>0&&<><b>{Math.round(shot?.distance??0)} m</b><small>{shot?.penalty?'Penalty · replay from previous lie':`${Math.round(shot?.carry??0)} m carry`}</small></>}</div>}
  {!busy&&!round.complete&&!cartMode&&<aside className={`mh-caddie ${caddieOpen?'expanded':'compact'}`} aria-label="Caddie recommendation"><div className="mh-caddie-head"><span><i/> CADDIE</span><button aria-label={caddieOpen?'Close caddie details':'Show caddie details'} aria-expanded={caddieOpen} aria-controls="mh-caddie-details" onClick={()=>setCaddieOpen(v=>!v)}>{caddieOpen?<X size={16}/>:<ChevronDown size={16}/>}</button></div>{advice?<><div className="mh-caddie-shot"><b>{CLUBS[advice.club].name}</b><span>{Math.round(advice.power*100)}%</span><span>{advice.club===13?`${advice.total.toFixed(1)} m roll`:`${Math.round(advice.carry)} m carry`}</span></div>{caddieOpen&&<p id="mh-caddie-details">{advice.reason}</p>}<button className="mh-use-shot" disabled={!allowed||swing.view.phase!=='ready'} onClick={caddie}>Use this shot <ArrowRight size={15}/></button></>:<p>{caddieError?'Set your club and line manually.':'Finding a safe line…'}</p>}</aside>}
  {cartMode&&<section className="mh-driving" aria-label="Cart controls"><div><span className="mh-eyebrow">TAKE THE SCENIC ROUTE</span><h2><output aria-label="Cart speed">{Math.round(cartSpeed*3.6)}</output><small> km/h</small></h2><p>{Math.round(cartDistance)} m from your ball<br/>WASD / arrows / left stick<br/>Hold Shift or R2 for boost</p></div><button className="drive-boost" aria-label="Toggle boost" aria-pressed={driveBoost} onClick={()=>setDriveBoost(v=>!v)}>⚡ Boost {driveBoost?'on':'off'}</button><DriveStick disabled={menu||help||card||map} onMove={onDriveTouch}/><button className="mh-use-shot" onClick={toggleCart}>Back to my ball <ArrowRight size={15}/></button></section>}
  {map&&<CourseMap current={1} onClose={closeMap}/>}
  <div className="mh-bottom-note"><span>{saveError?'Round save unavailable':gamepad?'CONTROLLER CONNECTED':'A / D AIM · SPACE SWING · C CLUB'}</span><span>MYHRA STUDY / 01</span></div>
  {bag&&<div className="mh-bag" role="dialog" aria-label="Golf bag"><div><span>YOUR BAG</span><button aria-label="Close golf bag" onClick={()=>setBag(false)}><X size={18}/></button></div>{CLUBS.map((c,i)=><button className={club===i?'selected':''} key={c.name} onClick={()=>{setClub(i);setBag(false);setPower(1);}}><span>{c.name}</span><small>{c.loft}°</small>{club===i&&<Check size={14}/>}</button>)}</div>}
  {(menu||help||card)&&<div className="mh-backdrop" onClick={()=>{setMenu(false);setHelp(false);setCard(false);}}><section ref={dialog} className="mh-dialog" role="dialog" aria-modal="true" aria-label={help?'How to play':card?'Scorecard':'Round settings'} onClick={e=>e.stopPropagation()}><button className="mh-close" aria-label="Close dialog" onClick={()=>{setMenu(false);setHelp(false);setCard(false);}}><X size={22}/></button><span className="mh-eyebrow">GRENLAND · MYHRA</span>
   {help?<><h2>Find your rhythm.</h2><p>Pick a club and set your line. Pull down on the swing circle, then smoothly return to the start. Backswing sets power; path and tempo shape the shot.</p><dl><dt>Mouse & touch</dt><dd>Use the swing circle. Tap the course to aim; drag to look around without changing your aim. Reset view lines the camera up behind your shot. The map lets you visit any hole.</dd><dt>Keyboard</dt><dd>Space starts the swing meter. Three more timed presses set power, line and tempo. A / D aim; arrows adjust power; C changes club; V scouts.</dd><dt>PS5 / standard controller</dt><dd>Right stick back and through. Left stick aims and sets power. L1 / R1 change club, X timing swing, Square scout, Options menu. Drive with the left stick and hold R2 to boost. Touch: lower-right circular joystick; Boost toggles extra speed.</dd></dl><p className="mh-small">Myhra uses Grenland terrain, the club’s hole guide and drone footage. The pond contour, path, tee details and pin remain approximations awaiting club verification.</p></>:card?<><h2>Your hole.</h2><div className="mh-score-total"><span>01 / MYHRA · PAR 4</span><strong>{round.strokes}<small> strokes</small></strong></div>{round.history.length?<ol className="mh-shot-list">{round.history.map((s,i)=><li key={i}><b>{i+1}</b><span>{s.club}<small>{s.lie}{s.penalty?' · penalty':''}</small></span><strong>{Math.round(s.distance)} m</strong></li>)}</ol>:<p>A fresh scorecard. Your first shot is waiting.</p>}</>:<><h2>Make it your round.</h2><label>Graphics<select value={quality} onChange={e=>setQuality(e.target.value as 'balanced'|'high')}><option value="balanced">Balanced · mobile</option><option value="high">High · desktop</option></select></label><label>Daylight <span><Sun size={14}/> {String(Math.floor(hour)).padStart(2,'0')}:{String(Math.floor((hour%1)*60)).padStart(2,'0')}</span></label><input className="mh-day-slider" aria-label="Time of day" type="range" min="6.5" max="19.5" step=".25" value={hour} onChange={e=>setHour(+e.target.value)}/><label>Let the day pass<input type="checkbox" checked={dayCycle} onChange={e=>setDayCycle(e.target.checked)}/></label><label>Forgiving swing<input type="checkbox" checked={friendly} onChange={e=>setFriendly(e.target.checked)}/></label><label>Swing controls<select value={swingMode} onChange={e=>setSwingMode(e.target.value as SwingMode)}><option value="analog">Analog swing</option><option value="three-click">Three-click timing</option></select></label><button className="mh-menu-row" onClick={()=>setSound(v=>!v)}>{sound?<Volume2 size={18}/>:<VolumeX size={18}/>} Sound {sound?'on':'off'}</button><button className="mh-menu-row" onClick={()=>{setCaddieOpen(true);setMenu(false);}}><Target size={18}/> Ask the caddie for a line</button><div className="mh-practice"><span>EXPLORE THE HOLE</span><button onClick={()=>practice(70)}>Approach · 70 m <ArrowRight size={15}/></button><button onClick={()=>practice(4)}>Putting · 4 m <ArrowRight size={15}/></button></div><button className="mh-menu-row" onClick={restart}><RotateCcw size={17}/> Restart at the tee</button><a className="mh-small" href="/play/grenland">Open the full-course preview <ArrowRight size={13}/></a></>}
  </section></div>}
 </main>;
}

export default function MyhraApp(){
 const [hole,setHole]=useState<Hole>(),[error,setError]=useState('');
 useEffect(()=>{const controller=new AbortController();loadHole(controller.signal).then(setHole).catch(e=>{if(!controller.signal.aborted)setError(String(e.message??e));});return()=>controller.abort();},[]);
 if(!hole)return <main className="myhra-game mh-loading"><span className="mh-emblem">G</span><h1>{error?'The course is taking a moment.':'A morning at Grenland.'}</h1><p>{error||'Loading Myhra · 329 m · Par 4'}</p>{error&&<button onClick={()=>location.reload()}>Try again</button>}<small>GRENLAND GOLF CLUB · NORWAY</small></main>;
 return <PlayHole hole={hole}/>;
}
