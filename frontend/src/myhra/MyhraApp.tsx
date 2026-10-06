import {lazy,Suspense,useCallback,useEffect,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {ArrowRight,CarFront,Check,ChevronDown,ChevronLeft,ChevronRight,Flag,HelpCircle,Menu,RotateCcw,Sun,Target,Volume2,VolumeX,Wind,X,BookOpen,ArrowLeft} from 'lucide-react';
import {CLUBS,bearingTo,distance,heightAt,scoreName,surfaceAt,type ShotResult,type Vec3} from '../course/engine';
import {useSwing} from '../course/useSwing';
import type {Strike} from '../course/swing';
import {IMPACT_DELAY} from './golfMotion';
import {golfSound} from '../course/audio';
import {loadHole,type Hole} from './data';
import {SAVE,LEGACY_SAVE,freshHole,restoreHole,recordHoleShot,courseRevision,type SavedHole} from './round';
import {ErrorBoundary} from '../components/shared/ErrorBoundary';
import type {Advice} from './caddie';
import {DriveStick} from '../course/DriveStick';
import {useDriveInput} from '../course/useDriveInput';
import CourseMap from '../course/CourseMap';
import MyhraSwingControl from './MyhraSwingControl';
import {PUTT_RANGES,recommendPuttRange,puttRangeMetres,simulateShot,type PuttRange,type ShotType} from './shot';
import {readSettings,SETTINGS_KEY,type Settings} from './settings';
import {saveWhatIfResult,type WhatIfContext} from './journal';
import type {JournalReplayFrame} from './RoundJournal';
import {illustrativeFlight} from './replay';
import './myhra.css';

const Scene=lazy(()=>import('./MyhraScene'));
const RoundJournal=lazy(()=>import('./RoundJournal'));
type View='player'|'overview'|'green';
type Panel='menu'|'help'|'score'|'bag'|'map'|'journal'|null;
type Session={kind:'practice'|'what-if';round:SavedHole;start:Vec3;context?:WhatIfContext;wind?:number};
type PerformanceSample={fps:number;p95:number;p99:number;triangles:number;drawCalls:number};
const clamp=(n:number,min:number,max:number)=>Math.max(min,Math.min(max,n));
const scoreValue=(strokes:number,par=4)=>strokes===par?'E':`${strokes>par?'+':''}${strokes-par}`;
const meters=(v:number)=>v<10?v.toFixed(1):Math.round(v).toString();
const isTextEntry=(target:EventTarget|null)=>(target as HTMLElement)?.closest('input,select,textarea,a,[contenteditable=true]');
function initialRound(hole:Hole){try{return restoreHole(hole,localStorage.getItem(SAVE)??localStorage.getItem(LEGACY_SAVE));}catch{return freshHole(hole);}}
function suggestedClub(hole:Hole,ball:Vec3){const lie=surfaceAt(hole.world,ball[0],ball[2]),d=distance(ball,hole.data.pin);return lie==='green'?13:lie==='bunker'&&d<70?11:d<55?11:d<100?9:d<135?7:d<170?5:d<205?2:0;}
function suggestedPower(hole:Hole,ball:Vec3){const d=distance(ball,hole.data.pin);return surfaceAt(hole.world,ball[0],ball[2])==='green'?clamp(d/puttRangeMetres(recommendPuttRange(d)),.01,.95):1;}

function MiniMap({hole,ball}:{hole:Hole;ball:Vec3}){
 const point=(p:number[])=>`${(p[0]-290)*.68},${(p[1]-195)*.43}`;
 const regions=hole.data.regions.filter(r=>r.points.some(p=>p[0]>330&&p[0]<460&&p[1]>220&&p[1]<590));
 return <svg viewBox="0 0 136 178" aria-label="Hole map" role="img"><rect width="136" height="178" rx="10" fill="#152e2d"/>{regions.map(r=><polygon key={r.id} points={r.points.map(point).join(' ')} fill={r.kind==='water'?'#74a3ad':r.kind==='green'?'#a7b991':r.kind==='bunker'?'#e1d7b7':'#4b6b4d'}/>)}<path d={`M ${point([ball[0],ball[2]])} L ${point([hole.data.pin[0],hole.data.pin[2]])}`} fill="none" stroke="#e6dbb8" opacity=".4" strokeDasharray="2 3"/><circle cx={(hole.data.pin[0]-290)*.68} cy={(hole.data.pin[2]-195)*.43} r="3" fill="#e6c577"/><circle cx={(ball[0]-290)*.68} cy={(ball[2]-195)*.43} r="4" fill="#fff" stroke="#20352b" strokeWidth="1.5"/></svg>;
}

function PlayHole({hole:sourceHole}:{hole:Hole}){
 const [settings,setSettings]=useState(readSettings);
 const [activeRound,setActiveRound]=useState<SavedHole>(()=>initialRound(sourceHole));
 const [session,setSession]=useState<Session|null>(null),round=session?.round??activeRound;
 const wind=session?.wind??settings.wind,setWind=useCallback((value:number)=>{if(session)setSession(s=>s?{...s,wind:value}:s);else setSettings(s=>({...s,wind:value}));},[session]);
 const hole=useMemo(()=>({...sourceHole,world:{...sourceHole.world,wind:[wind,0] as [number,number]}}),[sourceHole,wind]);
 const [club,setClub]=useState(()=>suggestedClub(hole,round.ball));
 const [power,setPower]=useState(()=>suggestedPower(hole,round.ball)),[aim,setAim]=useState(0),[view,setView]=useState<View>('player');
 const [shotType,setShotType]=useState<ShotType>('full');
 const [puttRange,setPuttRange]=useState<PuttRange>(()=>recommendPuttRange(distance(round.ball,hole.data.pin)));
 const [shot,setShot]=useState<ShotResult>(),[motion,setMotion]=useState(0),[busy,setBusy]=useState(false),[ready,setReady]=useState(false);
 const [panel,setPanel]=useState<Panel>(null),[cameraReset,setCameraReset]=useState(0),[feedback,setFeedback]=useState('');
 const [gamepad,setGamepad]=useState(false),[saveError,setSaveError]=useState(''),[caddieOpen,setCaddieOpen]=useState(false);
 const [guidance,setGuidance]=useState<{key:string;advice?:Advice;error?:boolean}>();
 const [cartSpeed,setCartSpeed]=useState(0),[cartDistance,setCartDistance]=useState(0),[cartMode,setCartMode]=useState(false);
 const [performanceSample,setPerformanceSample]=useState<PerformanceSample>();
 const [replayFrame,setReplayFrame]=useState<JournalReplayFrame|null>(null);
 const dialog=useRef<HTMLElement>(null),accepted=useRef(false),resetRejectedSwing=useRef<()=>void>(()=>{});
 const roundRef=useRef(round),sessionRef=useRef(session);
 const reviewRequired=!!round.needsCourseReview;
 const allowed=ready&&!busy&&!cartMode&&!round.complete&&!panel&&!reviewRequired;
 const allowedRef=useRef(allowed);
 useLayoutEffect(()=>{roundRef.current=round;sessionRef.current=session;allowedRef.current=allowed;},[round,session,allowed]);
 const {input:travelRef,boost:driveBoost,setBoost:setDriveBoost,onTouch:onDriveTouch}=useDriveInput(cartMode&&!panel);
 const ball=round.ball,pin=hole.data.pin,remaining=distance(ball,pin),lie=surfaceAt(hole.world,ball[0],ball[2]);
 const aimTarget:Vec3=useMemo(()=>session?.context?.target??(round.strokes===0&&distance(ball,hole.data.tee)<2?[355,heightAt(hole.world,355,370),370]:pin),[session?.context?.target,round.strokes,ball,hole,pin]);
 const bearing=bearingTo(ball,aimTarget)+aim*Math.PI/180;
 const adviceKey=`${ball.join(',')}:${wind}`,advice=guidance?.key===adviceKey?guidance.advice:undefined;
 const setSetting=useCallback(<K extends keyof Settings>(key:K,value:Settings[K])=>setSettings(s=>({...s,[key]:value})),[]);
 useEffect(()=>{try{localStorage.setItem(SETTINGS_KEY,JSON.stringify(settings));}catch{/* Storage failure must not interrupt a shot. */}},[settings]);
 useEffect(()=>{if(!settings.dayCycle)return;const timer=setInterval(()=>setSettings(s=>({...s,hour:s.hour>=19.5?6.5:s.hour+1/60})),1000);return()=>clearInterval(timer);},[settings.dayCycle]);
 useEffect(()=>{
  if(round.complete)return;
  const worker=new Worker(new URL('./caddie.worker.ts',import.meta.url),{type:'module'});
  worker.onmessage=e=>setGuidance({key:adviceKey,advice:e.data.advice,error:!!e.data.error});worker.onerror=()=>setGuidance({key:adviceKey,error:true});
  worker.postMessage({id:round.strokes,world:hole.world,ball});return()=>worker.terminate();
 },[hole,ball,round.strokes,round.complete,adviceKey]);
 const commit=useCallback((next:SavedHole)=>{
  if(!sessionRef.current){
   try{localStorage.setItem(SAVE,JSON.stringify(next));setSaveError('');}
   catch{setSaveError('Your browser could not save. This shot or change was not accepted; your previous round is safe. Free some browser storage and retry.');return false;}
  }
  roundRef.current=next;
  if(sessionRef.current)setSession(s=>s?{...s,round:next}:s);
  else setActiveRound(next);
  return true;
 },[]);
 const onStrike=useCallback((strike:Strike)=>{
  if(!allowedRef.current||accepted.current)return;accepted.current=true;
  const {launch,result}=simulateShot(hole.world,ball,{club,power,bearing,puttRange,shotType:club===13?'putt':shotType,friendly:settings.friendly},strike);
  const next=recordHoleShot(hole,roundRef.current,{club:CLUBS[club].name,lie,start:ball,launch,result,friendly:settings.friendly,conditions:{stimp:hole.world.stimp,wind:hole.world.wind}});
  const firstAlternative=!!sessionRef.current?.context&&roundRef.current.strokes===0;
  if(!commit(next)){accepted.current=false;resetRejectedSwing.current();return;}
  setShot(result);setMotion(n=>n+1);setBusy(true);setCaddieOpen(false);
  setFeedback(`${strike.label} · ${Math.round(strike.power*100)}% strength${Math.abs(strike.face)>.8?` · ${Math.abs(strike.face).toFixed(1)}° ${strike.face<0?'left':'right'}`:''}`);
  if(firstAlternative&&sessionRef.current?.context){try{saveWhatIfResult(sessionRef.current.context,{start:ball,club:CLUBS[club].name,launch,result,courseRevision:courseRevision(hole),conditions:{stimp:hole.world.stimp,wind:hole.world.wind}});}catch{setSaveError('This alternative could not be saved. Your original round is unchanged.');}}
 },[hole,ball,club,power,bearing,puttRange,settings.friendly,lie,commit,shotType]);
 const swing=useSwing(onStrike,allowed,settings.swingMode);
 const {reset:swingReset,cancel:swingCancel,press:swingPress,controllerMove,controller:swingController}=swing;
 useLayoutEffect(()=>{resetRejectedSwing.current=swingReset;},[swingReset]);
 const inputActive=swing.view.phase!=='ready'&&swing.view.phase!=='finish';
 const livePower=swing.view.phase==='ready'?power:swing.view.phase==='downswing'?swing.view.peak:swing.view.phase==='path'||swing.view.phase==='tempo'?swing.view.lockedPower:swing.view.amount;
 const preview=useMemo(()=>simulateShot(hole.world,ball,{club,power:livePower,bearing,puttRange,shotType:club===13?'putt':shotType,friendly:settings.friendly}).result,[hole,ball,club,livePower,bearing,puttRange,settings.friendly,shotType]);
 const resetPresentation=useCallback(()=>{
  swingReset();accepted.current=false;setShot(undefined);setBusy(false);setShotType('full');setAim(0);setView('player');setCaddieOpen(false);setCartMode(false);setFeedback('');setCameraReset(n=>n+1);
  travelRef.current={forward:0,turn:0,boost:false};
 },[swingReset,travelRef]);
 const settled=useCallback(()=>{
  setBusy(false);accepted.current=false;swingReset();setAim(0);
  const r=roundRef.current;if(r.complete){if(settings.sound)golfSound('cup');return;}
  const d=distance(r.ball,pin);setClub(suggestedClub(hole,r.ball));
  const range=recommendPuttRange(d);setPuttRange(range);setPower(suggestedPower(hole,r.ball));
 },[hole,pin,settings.sound,swingReset]);
 const sceneReady=useCallback(()=>setReady(true),[]);
 const impact=useCallback(()=>{if(settings.sound)golfSound('hit');},[settings.sound]);
 const aimTo=useCallback((point:Vec3)=>{if(!allowedRef.current||swingController.current.phase!=='ready')return;const a=bearingTo(ball,point)-bearingTo(ball,aimTarget);setAim(Math.atan2(Math.sin(a),Math.cos(a))*180/Math.PI);},[ball,aimTarget,swingController]);
 const openPanel=useCallback((next:Panel)=>{swingCancel();setCaddieOpen(false);setPanel(next);},[swingCancel]);
 const closePanel=useCallback(()=>setPanel(null),[]);
 const restart=()=>{if(!commit(freshHole(hole)))return;resetPresentation();setClub(0);setPower(1);setPanel(null);};
 const practice=(metres:number)=>{
  resetPresentation();const p:Vec3=metres===70?[365,0,pin[2]+68]:[pin[0],0,pin[2]+metres];p[1]=heightAt(hole.world,p[0],p[2]);
  setSession({kind:'practice',round:{...freshHole(hole),ball:p},start:p});setClub(metres<10?13:11);setPuttRange(recommendPuttRange(metres));setPower(metres===70?.9:metres/6);setPanel(null);
 };
 const returnToRound=()=>{resetPresentation();setSession(null);setClub(suggestedClub(hole,activeRound.ball));setPuttRange(recommendPuttRange(distance(activeRound.ball,pin)));setPower(suggestedPower(hole,activeRound.ball));};
 const tryShot=useCallback((position:Vec3,context:WhatIfContext)=>{
  resetPresentation();const p:Vec3=[position[0],heightAt(hole.world,position[0],position[2]),position[2]];
  setSession({kind:'what-if',round:{...freshHole(hole),ball:p},start:p,context});setClub(Math.max(0,CLUBS.findIndex(c=>c.name===context.club)));setPuttRange(recommendPuttRange(distance(p,pin)));setPower(1);setPanel(null);
 },[hole,pin,resetPresentation]);
 const applyAdvice=()=>{if(!advice)return;setShotType('full');setClub(advice.club);setPower(advice.power);if(advice.puttRange)setPuttRange(advice.puttRange);const a=advice.bearing-bearingTo(ball,aimTarget);setAim(Math.atan2(Math.sin(a),Math.cos(a))*180/Math.PI);setFeedback(`Caddie line set · match the ${Math.round(advice.power*100)}% marker`);setCaddieOpen(false);};
 const toggleCart=()=>{swingCancel();travelRef.current={forward:0,turn:0,boost:false};setCartMode(v=>!v);setView('player');setCaddieOpen(false);};
 const travelStatus=useCallback((speed:number,metres:number)=>{setCartSpeed(speed);setCartDistance(metres);},[]);
 useEffect(()=>{
  const cancel=()=>{swingCancel();travelRef.current={forward:0,turn:0,boost:false};};
  const visibility=()=>{if(document.hidden)cancel();};
  const key=(e:KeyboardEvent)=>{
   if(e.code==='Escape'){setPanel(null);setCaddieOpen(false);cancel();return;}
   if(panel||reviewRequired||isTextEntry(e.target)||cartMode||!allowed)return;
   if(e.code==='Space'&&(e.target as HTMLElement)?.closest('button'))return;
   if(e.code==='Space'){e.preventDefault();if(!e.repeat)swingPress();return;}
   if(swingController.current.phase!=='ready')return;
   if(['ArrowLeft','KeyA','ArrowRight','KeyD'].includes(e.code)){e.preventDefault();setAim(v=>clamp(v+(['ArrowLeft','KeyA'].includes(e.code)?-.5:.5),-180,180));}
   if(e.code==='ArrowUp'||e.code==='ArrowDown'){e.preventDefault();setPower(v=>clamp(v+(e.code==='ArrowUp'?.02:-.02),.01,1));}
   if(e.code==='KeyC'&&!e.repeat)setClub(v=>(v+1)%CLUBS.length);
   if(e.code==='KeyR'&&!e.repeat){setView('player');setCameraReset(v=>v+1);}
   if(e.code==='KeyV'&&!e.repeat)setView(v=>v==='player'?'overview':'player');
  };
  addEventListener('keydown',key);addEventListener('blur',cancel);addEventListener('orientationchange',cancel);document.addEventListener('visibilitychange',visibility);
  return()=>{removeEventListener('keydown',key);removeEventListener('blur',cancel);removeEventListener('orientationchange',cancel);document.removeEventListener('visibilitychange',visibility);};
 },[allowed,cartMode,panel,reviewRequired,swingCancel,swingPress,swingController,travelRef]);
 useEffect(()=>{
  if(!panel||panel==='map'||panel==='journal')return;
  const previous=document.activeElement as HTMLElement,node=dialog.current;node?.querySelector<HTMLElement>('button')?.focus();
  const trap=(e:KeyboardEvent)=>{if(e.key!=='Tab'||!node)return;const nodes=Array.from(node.querySelectorAll<HTMLElement>('button:not(:disabled),a,input,select'));const first=nodes[0],last=nodes[nodes.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus();}};
  node?.addEventListener('keydown',trap);return()=>{node?.removeEventListener('keydown',trap);previous?.focus();};
 },[panel]);
 const buttons=useRef<boolean[]>([]),hadPad=useRef(false),padNeutral=useRef(false);
 useEffect(()=>{
  let raf=0,last=0;
  const frame=(now:number)=>{
   const dt=Math.min((now-last)/1000,.05);last=now;const pad=Array.from(navigator.getGamepads?.()??[]).find(p=>p?.connected);
   if(!!pad!==hadPad.current){hadPad.current=!!pad;setGamepad(!!pad);padNeutral.current=false;buttons.current=[];}
   if(pad){
    if(!padNeutral.current)padNeutral.current=pad.axes.every(a=>Math.abs(a)<.15)&&pad.buttons.every(b=>!b.pressed);
    const fresh=(i:number)=>padNeutral.current&&pad.buttons[i]?.pressed&&!buttons.current[i];
    if(allowed&&padNeutral.current){
     if(swingController.current.phase==='ready'){
      if(Math.abs(pad.axes[0])>.2)setAim(v=>clamp(v+pad.axes[0]*dt*18,-180,180));
      if(Math.abs(pad.axes[1])>.2)setPower(v=>clamp(v-pad.axes[1]*dt*.35,.01,1));
      if(fresh(4))setClub(v=>(v+CLUBS.length-1)%CLUBS.length);if(fresh(5))setClub(v=>(v+1)%CLUBS.length);
     }
     controllerMove(pad.axes[3]??0,pad.axes[2]??0,true);
     if(fresh(0))swingPress();if(fresh(2))setView(v=>v==='player'?'overview':'player');
    }else controllerMove(0,0,false);
    if(fresh(9))openPanel(panel?null:'menu');if(fresh(1)){setPanel(null);swingCancel();}
    buttons.current=pad.buttons.map(b=>b.pressed);
   }else controllerMove(0,0,false);
   raf=requestAnimationFrame(frame);
  };raf=requestAnimationFrame(frame);return()=>cancelAnimationFrame(raf);
 },[allowed,controllerMove,swingController,swingPress,swingCancel,openPanel,panel]);
 const shownBall=busy&&shot?shot.path[0]:ball,shownRemaining=distance(shownBall,pin),shownLie=surfaceAt(hole.world,shownBall[0],shownBall[2]);
 const shownStroke=busy?round.strokes-(shot?.penalty??0):round.complete?round.strokes:round.strokes+1;
 const score=round.complete&&!busy?scoreValue(round.strokes,hole.data.par):round.strokes>hole.data.par?scoreValue(round.strokes,hole.data.par):'E';
 const locked=busy||inputActive||!ready,elevation=pin[1]-heightAt(hole.world,shownBall[0],shownBall[2]);
 const rangeMetres=PUTT_RANGES.find(r=>r.id===puttRange)?.metres??15;
 const replayStart=replayFrame?.start,replayEnd=replayFrame?.end,replayClub=replayFrame?.club;
 const replayResult=useMemo(()=>replayStart&&replayEnd?illustrativeFlight(hole.world,replayStart,replayEnd,replayClub==='Putter'):undefined,[hole.world,replayStart,replayEnd,replayClub]);
 const replayClubIndex=replayClub?Math.max(0,CLUBS.findIndex(c=>c.name===replayClub)):club;
 return <main className={`myhra-game ${busy?'mh-playing':''} ${cartMode?'mh-cart-mode':''} ${session?'mh-session':''} ${club===13?'mh-putting':''} ${replayFrame?'mh-replay-mode':''}`}>
  <div className="mh-world"><ErrorBoundary><Suspense fallback={<div className="mh-loading"><span className="mh-emblem">G</span><p>Preparing Myhra</p><small>A moment on the first tee.</small></div>}><Scene hole={hole} ball={replayStart??ball} bearing={replayStart&&replayEnd?bearingTo(replayStart,replayEnd):bearing} result={replayResult??shot} motion={motion} swing={swingController} club={replayClubIndex} shotType={replayFrame?'full':shotType} view={view} quality={settings.quality} onReady={sceneReady} onSettled={settled} onImpact={impact} onAim={aimTo} preview={preview} showPrediction={settings.prediction} showGreenGrid={settings.grid} replay={replayFrame?{time:IMPACT_DELAY+replayFrame.progress*3.5,view:'follow'}:undefined} busy={busy} hour={settings.hour} cartMode={cartMode&&!replayFrame} travel={travelRef} drivingAllowed={cartMode&&!panel} cameraReset={cameraReset} onTravel={travelStatus} onPerformance={setPerformanceSample}/></Suspense></ErrorBoundary></div>
  <div className="mh-vignette"/>
  <header className="mh-header"><a className="mh-wordmark" href="/play/grenland/myhra"><span className="mh-emblem">G<span>•</span></span><span>GRENLAND<small>MYHRA · HOLE 01</small></span></a><div className="mh-hole-title"><span>A ROUND AT HOME</span><h1>Myhra</h1></div><nav aria-label="Game menu"><button aria-label="Round journal" disabled={busy} onClick={()=>openPanel('journal')}><BookOpen size={20}/></button><button aria-label="Full course map" disabled={busy} onClick={()=>openPanel('map')}><Flag size={19}/></button><button aria-label="Menu" onClick={()=>openPanel('menu')}><Menu size={23}/></button></nav></header>
  <div className="mh-score-strip"><span className="mh-hole-number">01</span><span>PAR <b>{hole.data.par}</b></span><span>{hole.data.distance} <small>m</small></span><button aria-label="Scorecard" onClick={()=>openPanel('score')}><span>{round.complete?'SCORE':'SHOT'} <b>{shownStroke}</b></span><strong>{score}</strong></button></div>
  <div className="mh-wind"><Wind size={18}/><span>{wind?`${Math.abs(wind)} m/s ${wind<0?'←':'→'}`:'Calm'}<small>{String(Math.floor(settings.hour)).padStart(2,'0')}:{String(Math.floor(settings.hour%1*60)).padStart(2,'0')}</small></span></div>
  <aside className="mh-pin-info"><span><Flag size={13}/> To the pin</span><strong>{meters(shownRemaining)}<small>m</small></strong><p>{shownLie==='green'?'Green':shownLie==='bunker'?'Bunker':shownLie==='rough'?'Rough':distance(shownBall,hole.data.tee)<2?'Tee':'Fairway'}<i/>{shownRemaining<35?`${Math.round(Math.abs(elevation)*100)} cm`:`${Math.abs(elevation).toFixed(1)} m`} {elevation>=0?'↑':'↓'}</p></aside>
  {session&&<div className="mh-session-banner"><span>{session.kind==='practice'?'Practice · round safe':'What if · original unchanged'}</span><button onClick={returnToRound} disabled={busy}><ArrowLeft size={15}/> Return to round</button></div>}
  {!cartMode&&!busy&&<nav className="mh-tools" aria-label="Shot tools"><button aria-label={view==='player'?'Scout the hole':'Return to golfer'} aria-pressed={view!=='player'} onClick={()=>setView(v=>v==='player'?'overview':'player')}><Target size={18}/><span>{view==='player'?'Scout':'Player'}</span></button><button aria-label="Drive the cart" onClick={toggleCart}><CarFront size={18}/><span>Cart</span></button><button aria-label="Caddie" aria-expanded={caddieOpen} disabled={round.complete} onClick={()=>setCaddieOpen(v=>!v)}><HelpCircle size={18}/><span>Caddie</span></button>{remaining<40&&!round.complete&&<button aria-label="Read the green" aria-pressed={view==='green'} onClick={()=>setView(v=>v==='green'?'player':'green')}><Flag size={18}/><span>Green</span></button>}<button aria-label="Reset shot camera" title="Reset view · R" onClick={()=>{setView('player');setCameraReset(v=>v+1);}}><RotateCcw size={18}/></button></nav>}
  <button className="mh-minimap" aria-label="Open full course map" disabled={busy} onClick={()=>openPanel('map')}><MiniMap hole={hole} ball={shownBall}/><span>COURSE MAP ↗</span></button>
  {!cartMode&&(!round.complete||busy)?<footer className="mh-shot-hud">
   <section className="mh-club"><span className="mh-eyebrow">{club===13?'Read the line':'Choose your shot'}</span><button className="mh-club-button" aria-label="Choose club" disabled={locked} onClick={()=>openPanel('bag')}><strong>{CLUBS[club].name}</strong><ChevronDown size={20}/></button><span className="mh-carry"><b>{meters(preview.distance)} m</b><small>{club===13?'predicted roll':'predicted total'}</small></span>{club===13&&<label className="mh-range-label">Range<select aria-label="Putting range" disabled={locked} value={puttRange} onChange={e=>setPuttRange(e.target.value as PuttRange)}>{PUTT_RANGES.map(r=><option key={r.id} value={r.id}>{r.metres} m</option>)}</select></label>}{club!==13&&club>=3&&<label className="mh-range-label">Shot<select aria-label="Shot type" value={shotType} disabled={locked} onChange={e=>setShotType(e.target.value as ShotType)}><option value="full">Full</option><option value="pitch">Pitch</option><option value="chip">Chip</option></select></label>}</section>
   <section className="mh-shot-adjust"><label htmlFor="mh-power">Target strength <b>{Math.round(power*100)}%</b></label><input id="mh-power" aria-label="Target power" type="range" min="1" max="100" value={power*100} disabled={locked} onChange={e=>setPower(+e.target.value/100)}/><div className="mh-aim"><button aria-label="Aim left" disabled={locked} onClick={()=>setAim(v=>clamp(v-.5,-180,180))}><ChevronLeft size={20}/></button><span>{Math.abs(aim)<.1?'On line':`${Math.abs(aim).toFixed(1)}° ${aim<0?'left':'right'}`}</span><button aria-label="Aim right" disabled={locked} onClick={()=>setAim(v=>clamp(v+.5,-180,180))}><ChevronRight size={20}/></button></div><small>{club===13?`Full stroke = ${rangeMetres} m on a flat green`:'Pull to match the gold marker'}</small></section>
   <MyhraSwingControl swing={swing} mode={settings.swingMode} disabled={!allowed} busy={busy} target={power} distance={preview.distance}/>
  </footer>:!cartMode&&<section className="mh-result" aria-label="Hole completed"><span className="mh-eyebrow">{session?'PRACTICE COMPLETE':'MYHRA · HOLE COMPLETE'}</span><h2>{scoreName(round.strokes,hole.data.par)}</h2><p><strong>{round.strokes}</strong> strokes <span>{scoreValue(round.strokes,hole.data.par)}</span></p><button onClick={session?returnToRound:restart}><RotateCcw size={17}/>{session?'Return to saved round':'Play Myhra again'}</button><button className="mh-quiet" onClick={()=>openPanel('score')}>See your shots <ArrowRight size={16}/></button></section>}
  {feedback&&!busy&&!cartMode&&!round.complete&&<div className="mh-shot-recap" role="status"><span>{feedback}</span>{shot&&<small>{meters(shot.distance)} m total · {meters(shot.carry)} m carry{shot.penalty?' · penalty':''}</small>}</div>}
  {!busy&&!round.complete&&!cartMode&&!panel&&<aside className={`mh-caddie ${caddieOpen?'expanded':'compact'}`} aria-label="Caddie recommendation"><div className="mh-caddie-head"><span><i/> Caddie</span><button aria-label={caddieOpen?'Close caddie details':'Show caddie details'} aria-expanded={caddieOpen} onClick={()=>setCaddieOpen(v=>!v)}>{caddieOpen?<X size={18}/>:<ChevronDown size={18}/>}</button></div>{advice?caddieOpen?<><div className="mh-caddie-shot"><b>{CLUBS[advice.club].name}</b><span>{Math.round(advice.power*100)}%</span><span>{meters(advice.club===13?advice.total:advice.carry)} m {advice.club===13?'roll':'carry'}</span></div><p>{advice.reason}</p><button className="mh-use-shot" disabled={!allowed||inputActive} onClick={applyAdvice}>Use this shot <ArrowRight size={16}/></button></>:<button className="mh-use-shot" aria-label="Use this shot" disabled={!allowed||inputActive} onClick={applyAdvice}><span className="mh-caddie-summary"><b>{CLUBS[advice.club].name}</b><small>{Math.round(advice.power*100)}% · {meters(advice.club===13?advice.total:advice.carry)} m {advice.club===13?'roll':'carry'}</small></span><span className="mh-caddie-apply">Use <ArrowRight size={14}/></span></button>:<p>{guidance?.error?'Choose your club and line.':'Reading the hole…'}</p>}</aside>}
  {cartMode&&<section className="mh-driving" aria-label="Cart controls"><div><span className="mh-eyebrow">Explore Myhra</span><h2><output aria-label="Cart speed">{Math.round(cartSpeed*3.6)}</output><small> km/h</small></h2><p>{Math.round(cartDistance)} m from your ball<br/><span className="mh-desktop-only">WASD / arrows · Shift to boost</span></p></div><button className="drive-boost" aria-label="Toggle boost" aria-pressed={driveBoost} onClick={()=>setDriveBoost(v=>!v)}>Boost {driveBoost?'on':'off'}</button><DriveStick disabled={!!panel} onMove={onDriveTouch}/><button className="mh-use-shot" onClick={toggleCart}>Back to my ball <ArrowRight size={16}/></button></section>}
  {panel==='map'&&<CourseMap current={1} onClose={closePanel}/>}
  {panel==='journal'&&<Suspense fallback={<div className="mh-backdrop"><p>Opening your journal…</p></div>}><RoundJournal hole={hole} onClose={closePanel} onTryShot={tryShot} onReplayFrame={setReplayFrame}/></Suspense>}
  {saveError&&<div className="mh-save-error" role="alert">{saveError}</div>}
  <div className="mh-bottom-note"><span>{gamepad?'Controller connected':'A / D aim · Space timing swing · C club'}</span><span>GRENLAND / MYHRA</span></div>
  {reviewRequired&&<div className="mh-backdrop"><section className="mh-dialog" role="dialog" aria-modal="true" aria-label="Saved course review"><h2>Your saved position is safe.</h2><p>{round.migrationWarning??'This round was saved with a different course revision. Its score and position have been preserved.'}</p><p>Continue at the same location using the current playing surface, or start a fresh hole. The original browser save remains untouched.</p><button className="mh-use-shot" onClick={()=>commit({...round,needsCourseReview:false,migrationWarning:undefined,courseRevision:courseRevision(hole)})}>Continue at saved position</button><button className="mh-menu-row" onClick={restart}>Start a fresh hole</button></section></div>}
  {panel&&panel!=='map'&&panel!=='journal'&&<div className="mh-backdrop" onClick={closePanel}><section ref={dialog} className={`mh-dialog ${panel==='bag'?'mh-bag-sheet':''}`} role="dialog" aria-modal="true" aria-label={panel==='help'?'How to play':panel==='score'?'Scorecard':panel==='bag'?'Golf bag':'Round settings'} onClick={e=>e.stopPropagation()}><button className="mh-close" aria-label="Close dialog" onClick={closePanel}><X size={22}/></button><span className="mh-eyebrow">GRENLAND · MYHRA</span>
   {panel==='help'?<><h2>Find your rhythm.</h2><dl><dt>Touch</dt><dd>Pull down in the swing circle to set strength. Release to strike. Slide sideways outside the circle to cancel. Your starting position never changes sensitivity.</dd><dt>Aim & camera</dt><dd>Tap or drag the course to aim. Use two fingers to look around and pinch to zoom; on desktop, right-drag and scroll. Reset view returns behind your shot.</dd><dt>Mouse / controller</dt><dd>Pull back, then cross forward through the start. A controller stick returning to centre does not strike. L1 / R1 change club; Square scouts; Options opens settings.</dd><dt>Three-click / keyboard</dt><dd>Three presses: start the meter, set power, set accuracy. Space works with either control preference. A / D aim; arrows adjust the target marker; C changes club.</dd><dt>Putting</dt><dd>Choose a labelled range, read the downhill grid and start line, then pull to the desired distance. The full-stroke range is shown for a flat green; slopes change the actual roll.</dd><dt>Practice & replay</dt><dd>Practice never replaces your saved round. In the journal, remembered shot positions and flights are labelled approximate. An alternative shot creates its own result.</dd></dl><p className="mh-small">Course contours and pin locations include authored approximations. This build does not claim surveyed green accuracy.</p></>:panel==='bag'?<><h2>Your bag.</h2><div className="mh-bag-grid">{CLUBS.map((c,i)=><button className={club===i?'selected':''} key={c.name} onClick={()=>{setClub(i);setPanel(null);setPower(1);}}><span>{c.name}<small>{c.loft}° loft</small></span>{club===i&&<Check size={20}/>}</button>)}</div></>:panel==='score'?<><h2>{session?'Practice score.':'Your hole.'}</h2><div className="mh-score-total"><span>01 / MYHRA · PAR {hole.data.par}</span><strong>{round.strokes}<small> strokes</small></strong></div>{round.history.length?<ol className="mh-shot-list">{round.history.map((s,i)=><li key={i}><b>{i+1}</b><span>{s.club}<small>{s.lie}{s.penalty?' · penalty':''}</small></span><strong>{meters(s.distance)} m</strong></li>)}</ol>:<p>Your first shot is waiting.</p>}<button className="mh-use-shot" onClick={()=>openPanel('journal')}><BookOpen size={17}/> Open round journal</button></>:<><h2>Make it your round.</h2><div className="mh-settings-grid"><label>Graphics<select value={settings.quality} onChange={e=>setSetting('quality',e.target.value as Settings['quality'])}><option value="balanced">Balanced · iPhone 15 Pro target</option><option value="high">High · desktop</option><option value="economy">Reduced · older phones</option></select></label><label>Swing controls<select value={settings.swingMode} onChange={e=>setSetting('swingMode',e.target.value as Settings['swingMode'])}><option value="analog">Touch release / back-through</option><option value="three-click">Three-click timing</option></select></label></div><label className="mh-toggle">Forgiving strike<input type="checkbox" checked={settings.friendly} onChange={e=>setSetting('friendly',e.target.checked)}/></label><label className="mh-toggle">Projected shot / putting break<input type="checkbox" checked={settings.prediction} onChange={e=>setSetting('prediction',e.target.checked)}/></label><label className="mh-toggle">Putting grid<input type="checkbox" checked={settings.grid} onChange={e=>setSetting('grid',e.target.checked)}/></label><label>Daylight <span><Sun size={16}/>{String(Math.floor(settings.hour)).padStart(2,'0')}:{String(Math.floor(settings.hour%1*60)).padStart(2,'0')}</span><input aria-label="Time of day" type="range" min="6.5" max="19.5" step=".25" value={settings.hour} onChange={e=>setSetting('hour',+e.target.value)}/></label><label className="mh-toggle">Let the day pass<input type="checkbox" checked={settings.dayCycle} onChange={e=>setSetting('dayCycle',e.target.checked)}/></label><label>Crosswind · {wind} m/s<input aria-label="Crosswind" type="range" min="-5" max="5" step=".5" value={wind} onChange={e=>setWind(+e.target.value)}/></label><button className="mh-menu-row" onClick={()=>setSetting('sound',!settings.sound)}>{settings.sound?<Volume2 size={19}/>:<VolumeX size={19}/>} Sound {settings.sound?'on':'off'}</button><button className="mh-menu-row" onClick={()=>openPanel('help')}><HelpCircle size={19}/><span>How to play</span></button><button className="mh-menu-row" onClick={()=>openPanel('journal')}><BookOpen size={19}/> Round journal & real-round replay</button><div className="mh-practice"><span>Practice · your round stays saved</span><button onClick={()=>practice(70)}>Approach · 70 m <ArrowRight size={17}/></button><button onClick={()=>practice(4)}>Putting · 4 m <ArrowRight size={17}/></button>{session&&<button onClick={()=>{returnToRound();setPanel(null);}}>Return to saved round <ArrowRight size={17}/></button>}</div><button className="mh-menu-row" onClick={restart}><RotateCcw size={18}/> Restart this hole</button><a className="mh-small" href="/play/grenland">Explore holes 2–18 · earlier course preview <ArrowRight size={15}/></a>{performanceSample&&<details className="mh-diagnostics"><summary>Performance diagnostics</summary><p>{Math.round(performanceSample.fps)} fps · p95 {performanceSample.p95.toFixed(1)} ms · p99 {performanceSample.p99.toFixed(1)} ms<br/>{Math.round(performanceSample.triangles/1000)}k triangles · {performanceSample.drawCalls} draw calls</p><small>Current browser sample. Physical-device acceptance is reported separately.</small></details>}</>}
  </section></div>}
 </main>;
}

export default function MyhraApp(){
 const [hole,setHole]=useState<Hole>(),[error,setError]=useState('');
 useEffect(()=>{const controller=new AbortController();loadHole(controller.signal).then(setHole).catch(e=>{if(!controller.signal.aborted)setError(String(e.message??e));});return()=>controller.abort();},[]);
 if(!hole)return <main className="myhra-game mh-loading"><span className="mh-emblem">G</span><h1>{error?'The course is taking a moment.':'A morning at Grenland.'}</h1><p>{error||'Loading Myhra · 329 m · Par 4'}</p>{error&&<button onClick={()=>location.reload()}>Try again</button>}<small>GRENLAND GOLF CLUB · NORWAY</small></main>;
 return <PlayHole hole={hole}/>;
}
