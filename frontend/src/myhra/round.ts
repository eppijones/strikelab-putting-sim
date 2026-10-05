import {CLUBS, distance, sessionId, type Shot, type ShotResult, type Vec3} from '../course/engine.ts';
import type {Hole} from './data';
import type {ResolvedLaunch} from './shot';
import {createShotContext,createShotIntent,type ShotContext,type ShotIntent} from './contracts.ts';

export interface SavedHistory {
  club:string; distance:number; carry:number; lie:string; penalty:number;
  id?:string; kind?:'legacy-summary'|'simulated'; timestamp?:string;
  start?:Vec3; end?:Vec3; launch?:Shot|ResolvedLaunch; path?:Vec3[]; pathTimes?:number[]; duration?:number;
  conditions?:{stimp:number;wind:[number,number]};
  provenance?:'game'|'simulator'; courseRevision?:string;
  context?:ShotContext;intent?:ShotIntent;
  outcome?:{schemaVersion:1;physicsVersion:string;made:boolean;reason:string};
}
export interface SavedHole {
  version:1|2; ball:Vec3; strokes:number; complete:boolean; history:SavedHistory[];
  courseRevision?:string; needsCourseReview?:boolean; migrationWarning?:string;
}
export const LEGACY_SAVE='strikelab.myhra.hole.v2';
export const SAVE='strikelab.myhra.hole.v3';
const revisions=new WeakMap<Hole,string>();

/** A physics-geometry revision, independent of cosmetic course updates. */
export function courseRevision(hole:Hole):string {
  const known=revisions.get(hole); if(known)return known;
  let hash=2166136261;
  const add=(s:string)=>{for(let i=0;i<s.length;i++)hash=Math.imul(hash^s.charCodeAt(i),16777619);};
  add(JSON.stringify([hole.data.tee,hole.data.pin,hole.data.par,hole.data.terrain,hole.data.detail,hole.data.regions]));
  for(const tile of [hole.world.terrain,hole.world.detail])if(tile){
    const bytes=new Uint8Array(tile.heights.buffer,tile.heights.byteOffset,tile.heights.byteLength);
    for(let i=0;i<bytes.length;i++)hash=Math.imul(hash^bytes[i],16777619);
    for(let i=0;i<tile.lies.length;i++)hash=Math.imul(hash^tile.lies[i],16777619);
  }
  const result=`myhra-${(hash>>>0).toString(16)}`;revisions.set(hole,result);return result;
}
export function freshHole(hole:Hole):SavedHole {
  return {version:2,courseRevision:courseRevision(hole),ball:[...hole.data.tee],strokes:0,complete:false,history:[]};
}
const point=(p:unknown):p is Vec3=>Array.isArray(p)&&p.length===3&&p.every(n=>typeof n==='number'&&Number.isFinite(n))&&p[0]>=0&&p[0]<=768&&p[2]>=0&&p[2]<=768&&Math.abs(p[1])<=10000;
const pathPoint=(p:unknown):p is Vec3=>Array.isArray(p)&&p.length===3&&p.every(n=>typeof n==='number'&&Number.isFinite(n)&&Math.abs(n)<=10000);
const finiteRange=(n:unknown,min:number,max:number):n is number=>typeof n==='number'&&Number.isFinite(n)&&n>=min&&n<=max;
export function validLaunch(v:unknown):v is Shot {
  if(!v||typeof v!=='object')return false;
  const s=v as Shot;
  return finiteRange(s.speed,0,120)&&finiteRange(s.bearing,-Math.PI*100,Math.PI*100)&&finiteRange(s.launch,-90,90)&&finiteRange(s.spin,-20000,20000);
}
export function copyLaunch(value:Shot):Shot|ResolvedLaunch {
  if(!validLaunch(value))throw new Error('Invalid launch data.');
  const basic={speed:value.speed,bearing:value.bearing,launch:value.launch,spin:value.spin},v=value as Partial<ResolvedLaunch>;
  if(v.schemaVersion===undefined)return basic;
  if(v.schemaVersion!==1||v.units!=='metres-seconds-radians-bearing-degrees-launch-rpm'||![v.physicsVersion,v.calibrationId,v.inputProfile].every(s=>typeof s==='string'&&s.length>0&&s.length<=100)||!['mouse','touch','controller','timing','preview','measured','estimated'].includes(v.source??'')||!['full','pitch','chip','putt'].includes(v.shotType??'')||!(v.club===null||Number.isInteger(v.club)&&v.club!>=0&&v.club!<14)||!(v.power===null||finiteRange(v.power,0,1.08))||!finiteRange(v.contact,0,1)||!finiteRange(v.faceErrorDeg,-180,180)||v.puttRange!==undefined&&!['short','medium','long'].includes(v.puttRange))throw new Error('Invalid resolved launch provenance.');
  return {...basic,schemaVersion:1,units:v.units,physicsVersion:v.physicsVersion!,calibrationId:v.calibrationId!,inputProfile:v.inputProfile!,source:v.source!,club:v.club!,shotType:v.shotType!,power:v.power!,contact:v.contact,faceErrorDeg:v.faceErrorDeg,...(v.puttRange?{puttRange:v.puttRange}:{})};
}
export function restoreHole(hole:Hole,raw:string|null):SavedHole {
  try {
    if(!raw||raw.length>2_000_000)return freshHole(hole);
    const saved=JSON.parse(raw);
    if(![1,2].includes(saved?.version)||!point(saved.ball)||!Number.isInteger(saved.strokes)||saved.strokes<0||!Array.isArray(saved.history)||saved.history.length>500||typeof saved.complete!=='boolean')return freshHole(hole);
    if(saved.courseRevision!==undefined&&(typeof saved.courseRevision!=='string'||saved.courseRevision.length>100))return freshHole(hole);
    let strokes=0;
    const history:SavedHistory[]=[];
    for(const h of saved.history){
      if(!h||!CLUBS.some(c=>c.name===h.club)||!['rough','fairway','green','bunker','out','water'].includes(h.lie)||![0,1].includes(h.penalty)||![h.distance,h.carry].every(n=>finiteRange(n,0,2000)))return freshHole(hole);
      strokes+=1+h.penalty;
      const entry:SavedHistory={club:h.club,distance:h.distance,carry:h.carry,lie:h.lie,penalty:h.penalty};
      if(h.kind==='simulated'){
        if(typeof h.id!=='string'||h.id.length>100||!point(h.start)||!point(h.end)||!validLaunch(h.launch)||!Array.isArray(h.path)||h.path.length<1||h.path.length>1024||!h.path.every(pathPoint)||!finiteRange(h.duration,0,180)||!['game','simulator'].includes(h.provenance)||typeof h.courseRevision!=='string')return freshHole(hole);
        Object.assign(entry,{id:h.id,kind:'simulated',start:h.start,end:h.end,launch:copyLaunch(h.launch),path:h.path,duration:h.duration,provenance:h.provenance,courseRevision:h.courseRevision});
        if(h.pathTimes!==undefined){if(!Array.isArray(h.pathTimes)||h.pathTimes.length!==h.path.length||!h.pathTimes.every((n:unknown,i:number)=>finiteRange(n,0,180)&&(i===0||n>=h.pathTimes[i-1])))return freshHole(hole);entry.pathTimes=[...h.pathTimes];}
        if(h.timestamp!==undefined){if(typeof h.timestamp!=='string'||!Number.isFinite(Date.parse(h.timestamp)))return freshHole(hole);entry.timestamp=h.timestamp;}
        if(h.conditions!==undefined){if(!finiteRange(h.conditions.stimp,1,30)||!Array.isArray(h.conditions.wind)||h.conditions.wind.length!==2||!h.conditions.wind.every((n:unknown)=>finiteRange(n,-100,100)))return freshHole(hole);entry.conditions={stimp:h.conditions.stimp,wind:[...h.conditions.wind] as [number,number]};}
        if(h.context!==undefined){const c=h.context;if(c.schemaVersion!==1||c.playingSurface!=='myhra-authoritative-grid'||typeof c.courseRevision!=='string'||typeof c.physicsVersion!=='string'||!point(c.ballGround)||!point(c.pinGround)||!finiteRange(c.conditions?.stimpFeet,1,30)||!Array.isArray(c.conditions?.windMetresPerSecond)||c.conditions.windMetresPerSecond.length!==2||!c.conditions.windMetresPerSecond.every((n:unknown)=>finiteRange(n,-100,100)))return freshHole(hole);entry.context={schemaVersion:1,courseRevision:c.courseRevision,physicsVersion:c.physicsVersion,playingSurface:c.playingSurface,ballGround:[c.ballGround[0],c.ballGround[1],c.ballGround[2]],pinGround:[c.pinGround[0],c.pinGround[1],c.pinGround[2]],conditions:{stimpFeet:c.conditions.stimpFeet,windMetresPerSecond:[c.conditions.windMetresPerSecond[0],c.conditions.windMetresPerSecond[1]]}};}
        if(h.intent!==undefined){const i=h.intent;if(i.schemaVersion!==1||typeof i.assistance?.enabled!=='boolean'||!finiteRange(i.assistance?.faceScale,0,1)||!finiteRange(i.assistance?.contactProtection,0,1))return freshHole(hole);const l=copyLaunch(i.launch) as ResolvedLaunch;if(!('schemaVersion' in l))return freshHole(hole);entry.intent={schemaVersion:1,source:l.source,club:l.club,shotType:l.shotType,launch:l,assistance:{...i.assistance}};}
        if(h.outcome!==undefined){const o=h.outcome;if(o.schemaVersion!==1||typeof o.physicsVersion!=='string'||typeof o.made!=='boolean'||typeof o.reason!=='string'||o.reason.length>500)return freshHole(hole);entry.outcome={schemaVersion:1,physicsVersion:o.physicsVersion,made:o.made,reason:o.reason};}
      }else if(h.kind!==undefined&&h.kind!=='legacy-summary')return freshHole(hole);
      else if(saved.version===1||h.kind==='legacy-summary')entry.kind='legacy-summary';
      history.push(entry);
    }
    if(saved.strokes!==strokes)return freshHole(hole);
    const sameRevision=saved.courseRevision===courseRevision(hole);
    if(saved.complete&&(strokes===0||(sameRevision&&distance(saved.ball,hole.data.pin)>.15)))return freshHole(hole);
    const restored:SavedHole={version:2,ball:[...saved.ball] as Vec3,strokes,complete:saved.complete,history,courseRevision:saved.courseRevision??'legacy-unknown'};
    if(!sameRevision){
      restored.needsCourseReview=true;
      restored.migrationWarning=saved.version===1?'This older save contains shot summaries only. Its original position is preserved; start a new round to use the current course.':'The course geometry has changed. This saved round and its positions are preserved; start a new round for the current course.';
    }
    return restored;
  }catch{return freshHole(hole);}
}

export interface HoleShotInput {
  club:string;lie:string;start:Vec3;launch:Shot;result:ShotResult;
  conditions?:{stimp:number;wind:[number,number]};provenance?:'game'|'simulator';friendly?:boolean;
}
export function compactPath(path:Vec3[],max=512):Vec3[] {
  if(path.length<=max)return path.map(p=>[...p]);
  return Array.from({length:max},(_,i)=>[...path[Math.round(i*(path.length-1)/(max-1))]] as Vec3);
}
export function compactResult(result:ShotResult):ShotResult {
  const path=compactPath(result.path),pathTimes=result.pathTimes?path.map((_,i)=>result.pathTimes![Math.round(i*(result.path.length-1)/Math.max(1,path.length-1))]):undefined;
  return {...result,end:[...result.end],path,...(pathTimes?{pathTimes}:{})};
}
export function recordHoleShot(hole:Hole,round:SavedHole,input:HoleShotInput):SavedHole {
  const {result,start,launch}=input;
  if(round.complete||round.needsCourseReview)throw new Error('Start a new round before recording this shot.');
  if(!point(start)||!point(result.end)||!validLaunch(launch))throw new Error('The shot contains invalid coordinates or launch values.');
  const revision=courseRevision(hole);
  const reduced=compactResult(result);
  const entry:SavedHistory={id:sessionId(),kind:'simulated',timestamp:new Date().toISOString(),club:input.club,lie:input.lie,penalty:result.penalty,carry:result.carry,distance:result.distance,start:[...start],end:[...result.end],launch:copyLaunch(launch),path:reduced.path,...(reduced.pathTimes?{pathTimes:reduced.pathTimes}:{}),duration:result.duration,provenance:input.provenance??'game',courseRevision:revision};
  if(input.conditions)entry.conditions={stimp:input.conditions.stimp,wind:[...input.conditions.wind]};
  entry.context=createShotContext(hole.world,start,revision);entry.outcome={schemaVersion:1,physicsVersion:result.physicsVersion??entry.context.physicsVersion,made:result.made,reason:result.reason};
  if('schemaVersion' in launch)entry.intent=createShotIntent(launch as ResolvedLaunch,input.friendly??false);
  return {...round,version:2,courseRevision:revision,ball:[...result.end],strokes:round.strokes+1+result.penalty,complete:result.made,history:[...round.history,entry]};
}
