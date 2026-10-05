import {CLUBS, PHYSICS_VERSION, sessionId, type Shot, type ShotResult, type Vec3} from '../course/engine.ts';
import {compactResult, copyLaunch, courseRevision, validLaunch} from './round.ts';
import type {Hole} from './data';

export const JOURNAL_SAVE='strikelab.grenland.journal.v1';
export const MAX_JOURNAL_BYTES=4_000_000;
export const PREVIEW_PARS=[4,4,3,5,4,3,5,4,4,3,5,4,4,4,3,5,4,4];
export interface JournalPosition {point:Vec3;method:'manual-map';courseRevision:string}
export interface JournalShot {
  id:string;kind:'stroke'|'penalty';club:string|null;start:JournalPosition|null;end:JournalPosition|null;
  penaltyStrokes:number;note:string;
}
export interface JournalHole {number:number;par:number|null;strokes:number|null;putts:number|null}
export interface JournalRound {
  id:string;revision:number;title:string;date:string|null;tee:string|null;
  courseId:'grenland';courseRevision:string;historicalPar:number|null;parSource:'preview-assumed'|'user-entered';
  reportedTotal:number|null;source:'manual';holes:JournalHole[];myhraShots:JournalShot[];
  pin:{point:Vec3;method:'synthetic-preview';courseRevision:string};
}
export interface TryShotContext {
  journalId:string;shotId:string;recordRevision:number;club:string;target:Vec3|null;kind:'what-if';
}
export type WhatIfContext=TryShotContext;
export interface WhatIfShotResult {
  start:Vec3;club:string;launch:Shot;result:ShotResult;courseRevision:string;
  conditions?:{stimp:number;wind:[number,number]};
}
export interface WhatIfRecord {
  id:string;createdAt:string;kind:'what-if';parent:{journalId:string;shotId:string;recordRevision:number;shot:JournalShot};
  start:Vec3;club:string;launch:Shot;courseRevision:string;engineVersion:string;
  conditions:{stimp:number;wind:[number,number]};
  result:ShotResult;
}
export interface RoundJournalDocument {schema:'grenland.round-journal';version:1;rounds:JournalRound[];scenarios:WhatIfRecord[]}
export interface JournalStorage {getItem(key:string):string|null;setItem(key:string,value:string):void}
export const emptyJournal=():RoundJournalDocument=>({schema:'grenland.round-journal',version:1,rounds:[],scenarios:[]});
export function createJournalRound(hole:Hole,total:number|null=null):JournalRound {
  const revision=courseRevision(hole);
  return {id:sessionId(),revision:1,title:total===77?'Personal best · 77':'My Grenland round',date:null,tee:null,courseId:'grenland',courseRevision:revision,historicalPar:null,parSource:'preview-assumed',reportedTotal:total,source:'manual',holes:PREVIEW_PARS.map((par,i)=>({number:i+1,par,strokes:null,putts:null})),myhraShots:[],pin:{point:[...hole.data.pin],method:'synthetic-preview',courseRevision:revision}};
}
export const newJournalShot=():JournalShot=>({id:sessionId(),kind:'stroke',club:null,start:null,end:null,penaltyStrokes:0,note:''});

type Obj=Record<string,unknown>;
function object(v:unknown,where:string):Obj {if(!v||typeof v!=='object'||Array.isArray(v))throw new Error(`${where} must be an object.`);return v as Obj;}
function text(v:unknown,max:number,where:string,empty=false):string {if(typeof v!=='string'||v.length>max||(!empty&&!v.trim()))throw new Error(`${where} is invalid.`);return v;}
function id(v:unknown,where:string):string {const s=text(v,100,where);if(!/^[a-zA-Z0-9_-]+$/.test(s))throw new Error(`${where} is invalid.`);return s;}
function number(v:unknown,min:number,max:number,where:string,integer=false):number {if(typeof v!=='number'||!Number.isFinite(v)||v<min||v>max||(integer&&!Number.isInteger(v)))throw new Error(`${where} must be ${integer?'a whole number':'a number'} from ${min} to ${max}.`);return v;}
const optionalNumber=(v:unknown,min:number,max:number,where:string)=>v===null?null:number(v,min,max,where,true);
function vec(v:unknown,where:string,bounded=true):Vec3 {if(!Array.isArray(v)||v.length!==3)throw new Error(`${where} must contain three coordinates.`);return [number(v[0],bounded?0:-5000,bounded?768:5000,where),number(v[1],-10000,10000,where),number(v[2],bounded?0:-5000,bounded?768:5000,where)];}
function position(v:unknown,where:string):JournalPosition|null {if(v===null)return null;const p=object(v,where);if(p.method!=='manual-map')throw new Error(`${where}: unsupported position source.`);return {point:vec(p.point,where),method:'manual-map',courseRevision:text(p.courseRevision,100,where)};}
function shot(v:unknown,where:string):JournalShot {
  const s=object(v,where);if(s.kind!=='stroke'&&s.kind!=='penalty')throw new Error(`${where}: unsupported event type.`);
  if(s.club!==null&&!CLUBS.some(c=>c.name===s.club))throw new Error(`${where}: unknown club.`);
  const result:JournalShot={id:id(s.id,where),kind:s.kind,club:s.club as string|null,start:position(s.start,where),end:position(s.end,where),penaltyStrokes:number(s.penaltyStrokes,0,10,where,true),note:text(s.note,280,where,true)};
  if(result.kind==='penalty'&&(result.penaltyStrokes<1||result.club!==null||result.start!==null||result.end!==null))throw new Error(`${where}: penalty events add strokes without a ball flight.`);
  return result;
}
function unique(values:string[],where:string){if(new Set(values).size!==values.length)throw new Error(`${where} contains duplicate identifiers.`);}
function round(v:unknown):JournalRound {
  const r=object(v,'Round');if(r.courseId!=='grenland'||r.source!=='manual')throw new Error('This file is not a supported Grenland manual record.');
  if(r.parSource!=='preview-assumed'&&r.parSource!=='user-entered')throw new Error('Unknown par source.');
  if(r.date!==null){const date=text(r.date,10,'Date');if(!/^\d{4}-\d{2}-\d{2}$/.test(date)||!Number.isFinite(Date.parse(date))||new Date(date).toISOString().slice(0,10)!==date)throw new Error('Use a valid round date.');}
  if(!Array.isArray(r.holes)||r.holes.length!==18)throw new Error('A scorecard must contain exactly 18 holes.');
  const holes=r.holes.map((v,i)=>{const h=object(v,'Hole');if(h.number!==i+1)throw new Error('Scorecard holes must run from 1 to 18.');const result={number:i+1,par:optionalNumber(h.par,1,8,'Par'),strokes:optionalNumber(h.strokes,1,40,'Gross score'),putts:optionalNumber(h.putts,0,40,'Putts')};if(result.putts!==null&&result.strokes!==null&&result.putts>result.strokes)throw new Error(`Hole ${i+1}: putts exceed gross strokes.`);return result;});
  if(!Array.isArray(r.myhraShots)||r.myhraShots.length>80)throw new Error('A record can contain up to 80 Myhra events.');
  const shots=r.myhraShots.map((s,i)=>shot(s,`Event ${i+1}`));unique(shots.map(s=>s.id),'Shot list');
  const pin=object(r.pin,'Pin');if(pin.method!=='synthetic-preview')throw new Error('Unsupported pin provenance.');
  return {id:id(r.id,'Round ID'),revision:number(r.revision,1,1_000_000,'Record revision',true),title:text(r.title,80,'Title'),date:r.date as string|null,tee:r.tee===null?null:text(r.tee,40,'Tee'),courseId:'grenland',courseRevision:text(r.courseRevision,100,'Course revision'),historicalPar:optionalNumber(r.historicalPar,18,144,'Historical par'),parSource:r.parSource,reportedTotal:optionalNumber(r.reportedTotal,18,720,'Round total'),source:'manual',holes,myhraShots:shots,pin:{point:vec(pin.point,'Pin'),method:'synthetic-preview',courseRevision:text(pin.courseRevision,100,'Pin revision')}};
}
function conditions(v:unknown):{stimp:number;wind:[number,number]} {const c=object(v,'Conditions');if(!Array.isArray(c.wind)||c.wind.length!==2)throw new Error('Wind needs two components.');return {stimp:number(c.stimp,1,30,'Green speed'),wind:[number(c.wind[0],-100,100,'Wind'),number(c.wind[1],-100,100,'Wind')]};}
function scenario(v:unknown):WhatIfRecord {
  const s=object(v,'Scenario'),p=object(s.parent,'Scenario source'),r=object(s.result,'Scenario result');
  if(s.kind!=='what-if'||!validLaunch(s.launch))throw new Error('Unsupported scenario or launch.');
  if(typeof s.createdAt!=='string'||!Number.isFinite(Date.parse(s.createdAt)))throw new Error('Invalid scenario date.');
  if(!CLUBS.some(c=>c.name===s.club)||typeof r.made!=='boolean'||!Array.isArray(r.path)||r.path.length<1||r.path.length>512)throw new Error('Invalid scenario result.');
  const original=shot(p.shot,'Original shot');if(p.shotId!==original.id)throw new Error('Scenario source identifier does not match.');
  const result:ShotResult={end:vec(r.end,'Scenario end'),path:r.path.map((p,i)=>vec(p,`Flight ${i}`,false)),made:r.made,penalty:number(r.penalty,0,10,'Penalty',true),reason:text(r.reason,200,'Outcome',true),carry:number(r.carry,0,3000,'Carry'),distance:number(r.distance,0,3000,'Distance'),duration:number(r.duration,0,180,'Duration')};
  if(r.pathTimes!==undefined){if(!Array.isArray(r.pathTimes)||r.pathTimes.length!==result.path.length)throw new Error('Path timing does not match its samples.');const times=r.pathTimes.map((v,i)=>number(v,0,result.duration,`Path time ${i}`));if(times.some((n,i)=>i>0&&n<times[i-1]))throw new Error('Path times must increase.');result.pathTimes=times;}
  if(r.physicsVersion!==undefined)result.physicsVersion=text(r.physicsVersion,100,'Physics version');
  if(r.apex!==undefined)result.apex=number(r.apex,0,3000,'Apex');
  if(r.landing!==undefined)result.landing=vec(r.landing,'Landing',false);
  return {id:id(s.id,'Scenario ID'),createdAt:s.createdAt,kind:'what-if',parent:{journalId:id(p.journalId,'Source round'),shotId:id(p.shotId,'Source shot'),recordRevision:number(p.recordRevision,1,1_000_000,'Source revision',true),shot:original},start:vec(s.start,'Scenario start'),club:s.club as string,launch:copyLaunch(s.launch),courseRevision:text(s.courseRevision,100,'Scenario course'),engineVersion:text(s.engineVersion,100,'Engine version'),conditions:conditions(s.conditions),result};
}
export function parseJournal(raw:string):RoundJournalDocument {
  if(raw.length>MAX_JOURNAL_BYTES)throw new Error('The journal exceeds the 4 MB limit.');
  let data:unknown;try{data=JSON.parse(raw);}catch{throw new Error('The file is not valid JSON.');}
  const d=object(data,'Journal');if(d.schema!=='grenland.round-journal'||d.version!==1)throw new Error('Unsupported journal format/version. Export a Grenland round journal JSON file.');
  if(!Array.isArray(d.rounds)||d.rounds.length>30||!Array.isArray(d.scenarios)||d.scenarios.length>100)throw new Error('A journal supports 30 rounds and 100 what-if shots.');
  const rounds=d.rounds.map(round),scenarios=d.scenarios.map(scenario);unique(rounds.map(r=>r.id),'Journal');unique(scenarios.map(s=>s.id),'Scenarios');
  if(scenarios.some(s=>!rounds.some(r=>r.id===s.parent.journalId)))throw new Error('A scenario is missing its original round.');
  return {schema:'grenland.round-journal',version:1,rounds,scenarios};
}
export function readJournal(storage:JournalStorage=localStorage):RoundJournalDocument {const raw=storage.getItem(JOURNAL_SAVE);return raw?parseJournal(raw):emptyJournal();}
/** Validate the complete candidate, then replace one key. A failed write leaves the prior document intact. */
export function writeJournal(doc:RoundJournalDocument,storage:JournalStorage=localStorage):RoundJournalDocument {
  const serialized=JSON.stringify(doc,(_key,value)=>{if(typeof value==='number'&&!Number.isFinite(value))throw new Error('The journal contains a non-finite number.');return value;});
  const canonical=parseJournal(serialized);storage.setItem(JOURNAL_SAVE,JSON.stringify(canonical));return canonical;
}
export function saveWhatIfResult(context:TryShotContext,input:WhatIfShotResult,storage:JournalStorage=localStorage):WhatIfRecord {
  const doc=readJournal(storage),record=doc.rounds.find(r=>r.id===context.journalId),source=record?.myhraShots.find(s=>s.id===context.shotId);
  if(!record||!source||record.revision!==context.recordRevision)throw new Error('The original record changed. Reopen it before starting another what-if.');
  if(doc.scenarios.length>=100)throw new Error('The journal has 100 scenarios. Export it and remove an older scenario first.');
  const saved:WhatIfRecord={id:sessionId(),createdAt:new Date().toISOString(),kind:'what-if',parent:{journalId:record.id,shotId:source.id,recordRevision:record.revision,shot:structuredClone(source)},start:[...input.start],club:input.club,launch:copyLaunch(input.launch),courseRevision:input.courseRevision,engineVersion:input.result.physicsVersion??PHYSICS_VERSION,conditions:input.conditions??{stimp:10,wind:[0,0]},result:compactResult(input.result)};
  const validated=writeJournal({...doc,scenarios:[...doc.scenarios,saved]},storage);return validated.scenarios[validated.scenarios.length-1];
}
export function roundSummary(r:JournalRound) {
  const entered=r.holes.filter(h=>h.strokes!==null),sum=entered.reduce((n,h)=>n+(h.strokes??0),0),full=entered.length===18;
  const parKnown=entered.every(h=>h.par!==null),relative=entered.length&&parKnown?entered.reduce((n,h)=>n+(h.strokes??0)-(h.par??0),0):null;
  const myhraCount=r.myhraShots.reduce((n,s)=>n+(s.kind==='stroke'?1:0)+s.penaltyStrokes,0);
  const warnings:string[]=[];
  if(full&&r.reportedTotal!==null&&sum!==r.reportedTotal)warnings.push(`Scorecard totals ${sum}; reported total is ${r.reportedTotal}. Both are retained until corrected.`);
  if(!full&&r.reportedTotal!==null&&sum>=r.reportedTotal&&entered.length)warnings.push('Entered holes already meet or exceed the reported total; the scorecard is incomplete.');
  if(r.myhraShots.length&&r.holes[0].strokes!==null&&myhraCount!==r.holes[0].strokes)warnings.push(`Myhra events total ${myhraCount}; hole 1 score is ${r.holes[0].strokes}. Check missing shots or penalties.`);
  const holePars=r.holes.every(h=>h.par!==null)?r.holes.reduce((n,h)=>n+(h.par??0),0):null;
  if(r.historicalPar!==null&&holePars!==null&&r.historicalPar!==holePars)warnings.push(`Hole pars total ${holePars}; historical course par is ${r.historicalPar}.`);
  for(let i=1;i<r.myhraShots.length;i++){
    const a=r.myhraShots[i-1],b=r.myhraShots[i];
    if(a.kind==='stroke'&&b.kind==='stroke'&&a.end&&b.start&&Math.hypot(a.end.point[0]-b.start.point[0],a.end.point[2]-b.start.point[2])>3)warnings.push(`Events ${i} and ${i+1} have a gap between positions; confirm the lie or add a penalty/drop note.`);
  }
  return {entered:entered.length,total:sum,full,relative,myhraCount,warnings,holePars};
}
export const scoreRelative=(v:number)=>v===0?'E':`${v>0?'+':''}${v}`;
export function illustrativePosition(s:JournalShot,t:number):Vec3|null {
  if(s.kind!=='stroke'||!s.start||!s.end)return null;
  const p=Math.max(0,Math.min(1,t)),a=s.start.point,b=s.end.point;
  const span=Math.hypot(a[0]-b[0],a[2]-b[2]);
  // This is presentation only; never infer observed carry, launch, or trajectory from endpoints.
  const apex=s.club==='Putter'?0:Math.min(25,span*.12);
  return [a[0]+(b[0]-a[0])*p,a[1]+(b[1]-a[1])*p+Math.sin(Math.PI*p)*apex,a[2]+(b[2]-a[2])*p];
}
