import test from 'node:test';
import assert from 'node:assert/strict';
import {createJournalRound,emptyJournal,illustrativePosition,newJournalShot,parseJournal,readJournal,roundSummary,saveWhatIfResult,writeJournal,JOURNAL_SAVE} from '../src/myhra/journal.ts';
import {courseRevision,freshHole,restoreHole,recordHoleShot,SAVE,LEGACY_SAVE,compactResult} from '../src/myhra/round.ts';
import {resolveLaunch} from '../src/myhra/shot.ts';

function makeHole(){const tile={url:'test',surfaces:'test',size:3,spacing:384,origin:[0,0],heights:new Float32Array(9),lies:new Uint8Array(9).fill(1)};return {data:{name:'Myhra',par:4,distance:329,tee:[384,0,570],pin:[384,0,241],terrain:tile,detail:tile,regions:[],trees:[],pond:{points:[],height:0},road:[]},world:{terrain:tile,pin:[384,0,241],stimp:10,wind:[0,0]}};}
const memory=()=>{const data=new Map();return {getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,v)};};
const hole=makeHole(),record=createJournalRound(hole,77),document={...emptyJournal(),rounds:[record]};
const position=p=>({point:p,method:'manual-map',courseRevision:courseRevision(hole)});
const shot={...newJournalShot(),club:'Driver',start:position([384,0,570]),end:position([370,0,390])};
const result={end:[370,0,395],path:[[384,.02135,570],[375,12,480],[370,.02135,395]],pathTimes:[0,2,5],made:false,penalty:0,reason:'Stopped',carry:165,distance:175,duration:5,physicsVersion:'grenland-golf-2',apex:12};

test('77 alone remains a summary, with no invented hole scores or shots',()=>{
 const summary=roundSummary(record);assert.equal(summary.entered,0);assert.equal(summary.relative,null);assert.equal(summary.full,false);assert.equal(record.reportedTotal,77);assert.equal(record.historicalPar,null);assert.equal(record.myhraShots.length,0);assert.equal(summary.holePars,72);
 assert.deepEqual(parseJournal(JSON.stringify(document)),document);
});
test('scorecard, historical total, pars and stroke events are reconciled independently',()=>{
 const r=structuredClone(record);r.holes.forEach(h=>h.strokes=h.par);r.myhraShots=[shot];r.historicalPar=71;
 const s=roundSummary(r);assert.equal(s.total,72);assert.equal(s.relative,0);assert.equal(s.myhraCount,1);assert.equal(s.warnings.length,3);
 r.holes[0].strokes=9;r.reportedTotal=77;r.historicalPar=72;r.myhraShots=[];
 const complete=roundSummary(r);assert.equal(complete.relative,5);assert.equal(complete.total,77);assert.deepEqual(complete.warnings,[]);
});
test('illustrative playback retains entered endpoints and penalties never become flights',()=>{
 assert.deepEqual(illustrativePosition(shot,0),shot.start.point);
 const end=illustrativePosition(shot,1);assert.ok(end.every((v,i)=>Math.abs(v-shot.end.point[i])<1e-9));
 assert.ok(illustrativePosition(shot,.5)[1]>0);
 const putt={...shot,club:'Putter'};assert.equal(illustrativePosition(putt,.5)[1],0);
 assert.equal(illustrativePosition({...shot,kind:'penalty'},.5),null);
});
test('import rejects unsupported versions, bad coordinates, dates, score inflation and duplicate IDs',()=>{
 const invalid=[];
 invalid.push({...document,version:2});
 const mutate=f=>{const d=structuredClone(document);f(d);invalid.push(d);};
 mutate(d=>d.rounds[0].date='2026-02-30');
 mutate(d=>d.rounds[0].holes[0].strokes=1000);
 mutate(d=>d.rounds[0].holes[0].strokes='4');
 mutate(d=>{d.rounds[0].holes[0].strokes=4;d.rounds[0].holes[0].putts=5;});
 mutate(d=>d.rounds[0].holes.pop());
 mutate(d=>d.rounds.push(d.rounds[0]));
 mutate(d=>d.rounds[0].myhraShots=[{...shot,start:position([999,0,5])}]);
 mutate(d=>d.rounds[0].myhraShots=[shot,shot]);
 mutate(d=>d.rounds[0].myhraShots=[{...shot,kind:'penalty',penaltyStrokes:1}]);
 for(const d of invalid)assert.throws(()=>parseJournal(JSON.stringify(d)));
 assert.throws(()=>parseJournal('{broken'));
 assert.throws(()=>parseJournal(JSON.stringify(document).replace('"reportedTotal":77','"reportedTotal":1e999')));
 assert.throws(()=>parseJournal(' '.repeat(4_000_001)));
});
test('import canonicalizes extra properties without evaluating their content',()=>{
 const hostile=JSON.stringify(document).replace('"title":','"__proto__":{"polluted":true},"title":');
 const clean=parseJournal(hostile);assert.equal({}.polluted,undefined);assert.equal(Object.hasOwn(clean.rounds[0],'__proto__'),false);
 clean.rounds[0].title='<img src=x onerror=alert(1)>';
 assert.equal(parseJournal(JSON.stringify(clean)).rounds[0].title,clean.rounds[0].title);
});
test('journal validation and failed quota writes preserve the previous stored record',()=>{
 const storage=memory();writeJournal(document,storage);const before=storage.getItem(JOURNAL_SAVE);
 assert.throws(()=>writeJournal({...document,version:99},storage));assert.equal(storage.getItem(JOURNAL_SAVE),before);
 assert.throws(()=>writeJournal({...document,rounds:[{...record,reportedTotal:Infinity}]},storage),/non-finite/);assert.equal(storage.getItem(JOURNAL_SAVE),before);
 const full={getItem:storage.getItem,setItem:()=>{throw new Error('quota');}};
 assert.throws(()=>writeJournal({...document,rounds:[]},full),/quota/);assert.equal(storage.getItem(JOURNAL_SAVE),before);
});
test('what-if saves measured inputs and original snapshot without rewriting history',()=>{
 const storage=memory(),r={...record,myhraShots:[shot]},doc={...document,rounds:[r]};writeJournal(doc,storage);
 const launch=resolveLaunch(hole.world,shot.start.point,{club:0,power:.9,bearing:0},{power:.9,face:1,contact:1,tempo:0,label:'test',source:'touch'});
 const context={journalId:r.id,shotId:shot.id,recordRevision:r.revision,club:'Driver',target:shot.end.point,kind:'what-if'};
 const saved=saveWhatIfResult(context,{start:shot.start.point,club:'Driver',launch,result,courseRevision:courseRevision(hole)},storage);
 const after=readJournal(storage);assert.deepEqual(after.rounds,doc.rounds);assert.deepEqual(saved.parent.shot,shot);assert.equal(after.scenarios.length,1);assert.equal(saved.launch.source,'touch');assert.deepEqual(saved.result.pathTimes,[0,2,5]);
 assert.throws(()=>saveWhatIfResult({...context,recordRevision:2},{start:shot.start.point,club:'Driver',launch,result,courseRevision:courseRevision(hole)},storage),/changed/);
});
test('new simulation saves retain launch provenance, path timing and geometry revision',()=>{
 const launch=resolveLaunch(hole.world,shot.start.point,{club:0,power:1,bearing:0});
 const fresh=freshHole(hole),saved=recordHoleShot(hole,fresh,{club:'Driver',lie:'fairway',start:shot.start.point,launch,result});
 assert.equal(saved.history[0].launch.physicsVersion,'grenland-golf-2');assert.deepEqual(saved.history[0].pathTimes,[0,2,5]);
 const restored=restoreHole(hole,JSON.stringify(saved));assert.deepEqual(restored,saved);assert.notEqual(SAVE,LEGACY_SAVE);
});
test('legacy and changed geometry saves retain original positions and require review',()=>{
 const legacy={version:1,ball:[380,12,370],strokes:1,complete:false,history:[{club:'Driver',distance:200,carry:185,lie:'fairway',penalty:0}]};
 const migrated=restoreHole(hole,JSON.stringify(legacy));assert.deepEqual(migrated.ball,legacy.ball);assert.equal(migrated.needsCourseReview,true);assert.equal(migrated.history[0].kind,'legacy-summary');assert.equal(migrated.history[0].path,undefined);
 const changed=makeHole();changed.data.pin=[380,0,242];const saved={...freshHole(hole),ball:[380,12,370]};
 const restored=restoreHole(changed,JSON.stringify(saved));assert.deepEqual(restored.ball,saved.ball);assert.equal(restored.needsCourseReview,true);
});
test('compacting paths retains their matching times and exact final sample',()=>{
 const long={...result,path:Array.from({length:2000},(_,i)=>[384,0,570-i*.1]),pathTimes:Array.from({length:2000},(_,i)=>i/100),duration:20};
 const compact=compactResult(long);assert.equal(compact.path.length,512);assert.equal(compact.pathTimes.length,512);
 assert.deepEqual(compact.path.at(-1),long.path.at(-1));assert.equal(compact.pathTimes.at(-1),long.pathTimes.at(-1));
 for(let i=0;i<compact.path.length;i++)assert.ok(Math.abs((570-compact.path[i][2])*.1-compact.pathTimes[i])<1e-8);
});
