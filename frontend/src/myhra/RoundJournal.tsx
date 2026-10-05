import {useCallback,useEffect,useMemo,useRef,useState,type ChangeEvent,type KeyboardEvent} from 'react';
import {CLUBS,heightAt,type Vec3} from '../course/engine';
import type {Hole} from './data';
import {courseRevision} from './round';
import {JOURNAL_SAVE,MAX_JOURNAL_BYTES,createJournalRound,emptyJournal,illustrativePosition,newJournalShot,parseJournal,readJournal,roundSummary,scoreRelative,writeJournal,type JournalRound,type JournalShot,type RoundJournalDocument,type TryShotContext} from './journal';
import './RoundJournal.css';

export interface JournalReplayFrame {position:Vec3;start:Vec3;end:Vec3;progress:number;club:string;illustrative:true}
export interface RoundJournalProps {
  hole:Hole;onClose:()=>void;
  onTryShot?:(position:Vec3,context:TryShotContext)=>void;
  onReplayFrame?:(frame:JournalReplayFrame|null)=>void;
}
const optionalInteger=(s:string)=>s===''?null:Number(s);
const formatPoint=(p:Vec3)=>`${p[0].toFixed(1)}, ${p[2].toFixed(1)} m`;
const errorText=(error:unknown)=>error instanceof Error?error.message:'The journal could not be saved.';
function download(text:string,name:string){
  const url=URL.createObjectURL(new Blob([text],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

export default function RoundJournal({hole,onClose,onTryShot,onReplayFrame}:RoundJournalProps){
  const [initial]=useState(()=>{
    try {const saved=readJournal();return {doc:saved.rounds.length?saved:{...saved,rounds:[createJournalRound(hole,77)]},error:''};}
    catch(error){return {doc:{...emptyJournal(),rounds:[createJournalRound(hole,77)]},error:errorText(error)};}
  });
  const [doc,setDoc]=useState(initial.doc),[selectedId,setSelectedId]=useState(initial.doc.rounds[0].id);
  const [tab,setTab]=useState<'card'|'shots'|'scenarios'>('card');
  const [error,setError]=useState(initial.error),[notice,setNotice]=useState(''),[blocked,setBlocked]=useState(!!initial.error);
  const [shotId,setShotId]=useState<string|null>(null),[place,setPlace]=useState<'start'|'end'|null>(null);
  const [playing,setPlaying]=useState(false),[progress,setProgress]=useState(0),[wideMap,setWideMap]=useState(false),[cinema,setCinema]=useState(false);
  const [cursor,setCursor]=useState<Vec3>([...hole.data.tee]),[pendingImport,setPendingImport]=useState<RoundJournalDocument|null>(null);
  const dialog=useRef<HTMLElement>(null),file=useRef<HTMLInputElement>(null),map=useRef<SVGSVGElement>(null);
  const record=doc.rounds.find(r=>r.id===selectedId)??doc.rounds[0],summary=roundSummary(record);
  const selected=record.myhraShots.find(s=>s.id===shotId)??record.myhraShots[0];
  const revision=courseRevision(hole),sameCourse=record.courseRevision===revision;
  const playable=!!selected?.start&&!!selected?.end&&selected.kind==='stroke';
  const ball=selected?illustrativePosition(selected,progress):null;
  const scenarios=doc.scenarios.filter(s=>s.parent.journalId===record.id);
  const mapBackground=useMemo(()=><>
    <rect x={0} y={0} width={768} height={768} fill="#163d32"/>
    {hole.data.regions.map(r=><polygon key={r.id} points={r.points.map(p=>p.join(',')).join(' ')} fill={r.kind==='green'?'#b3c995':r.kind==='fairway'?'#658459':r.kind==='bunker'?'#d5c397':r.kind==='water'?'#518894':'#365b3c'} opacity={r.kind==='rough'?.65:1}/>)}
    <polygon points={hole.data.pond.points.map(p=>p.join(',')).join(' ')} fill="#5a92a1"/>
    <polyline points={hole.data.road.map(p=>`${p[0]},${p[2]}`).join(' ')} stroke="#b5ab91" strokeWidth={2} fill="none" opacity={.65}/>
    <circle cx={hole.data.tee[0]} cy={hole.data.tee[2]} r={4} fill="#f2db93"/><text x={hole.data.tee[0]+7} y={hole.data.tee[2]+3} fill="#fff5d8" fontSize={8}>Preview tee</text>
  </>,[hole]);

  useEffect(()=>{
    const prior=document.activeElement as HTMLElement|null;dialog.current?.querySelector<HTMLButtonElement>('button')?.focus();
    return()=>prior?.focus();
  },[]);
  useEffect(()=>{if(place){map.current?.scrollIntoView({block:'center',behavior:'instant'});map.current?.focus({preventScroll:true});}},[place]);
  const persist=useCallback((next:RoundJournalDocument)=>{
    if(blocked){setError('The existing journal could not be read. Export its backup and start a new journal before making changes.');return false;}
    try {const saved=writeJournal(next);setDoc(saved);setError('');setNotice('Saved on this device');return true;}
    catch(e){setError(errorText(e));return false;}
  },[blocked]);
  const edit=(change:Partial<JournalRound>)=>{
    const updated={...record,...change,revision:record.revision+1};
    return persist({...doc,rounds:doc.rounds.map(r=>r.id===record.id?updated:r)});
  };
  const changeShot=(change:Partial<JournalShot>)=>{if(selected)edit({myhraShots:record.myhraShots.map(s=>s.id===selected.id?{...s,...change}:s)});};
  const chooseShot=(id:string)=>{setShotId(id);setProgress(0);setPlaying(false);setPlace(null);};
  const chooseRound=(id:string)=>{setSelectedId(id);setShotId(null);setProgress(0);setPlaying(false);setPlace(null);};

  useEffect(()=>{
    if(!playing)return;
    let raf=0,last=0;
    const frame=(now:number)=>{const dt=last?Math.min((now-last)/1000,.1):0;last=now;setProgress(p=>Math.min(1,p+dt/3.5));raf=requestAnimationFrame(frame);};
    raf=requestAnimationFrame(frame);return()=>cancelAnimationFrame(raf);
  },[playing]);
  useEffect(()=>{if(progress>=1)setPlaying(false);},[progress]);
  const replayFrame=useMemo<JournalReplayFrame|null>(()=>{
    if(!cinema||tab!=='shots'||!selected?.start||!selected?.end||selected.kind!=='stroke'||!sameCourse)return null;
    const position=illustrativePosition(selected,progress);if(!position)return null;
    return {position,start:selected.start.point,end:selected.end.point,progress,club:selected.club??'Unknown club',illustrative:true};
  },[cinema,tab,selected,progress,sameCourse]);
  useEffect(()=>{onReplayFrame?.(replayFrame);},[replayFrame,onReplayFrame]);
  useEffect(()=>()=>onReplayFrame?.(null),[onReplayFrame]);

  const placeAt=(p:Vec3)=>{
    if(!place||!selected||!sameCourse)return;
    const point:Vec3=[Math.max(0,Math.min(768,p[0])),0,Math.max(0,Math.min(768,p[2]))];point[1]=heightAt(hole.world,point[0],point[2]);
    changeShot({[place]:{point,method:'manual-map',courseRevision:revision}});setCursor(point);setPlace(null);setPlaying(false);setProgress(0);
  };
  const mapKey=(e:KeyboardEvent<SVGSVGElement>)=>{
    const delta=e.shiftKey?1:5;
    if(e.key==='Enter'||e.key===' '){e.preventDefault();placeAt(cursor);return;}
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;
    e.preventDefault();setCursor(p=>[Math.max(0,Math.min(768,p[0]+(e.key==='ArrowRight'?delta:e.key==='ArrowLeft'?-delta:0))),p[1],Math.max(0,Math.min(768,p[2]+(e.key==='ArrowDown'?delta:e.key==='ArrowUp'?-delta:0)))]);
  };
  const keyDown=(e:KeyboardEvent<HTMLElement>)=>{
    e.stopPropagation();
    if(e.key==='Escape'){e.preventDefault();if(place)setPlace(null);else onClose();}
    if(e.key==='Tab'){
      const nodes=Array.from(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled):not([type="file"]),select:not(:disabled),textarea:not(:disabled),[tabindex="0"]')??[]).filter(n=>n.getClientRects().length);
      const first=nodes[0],last=nodes[nodes.length-1];
      if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus();}
    }
  };
  const importFile=async(e:ChangeEvent<HTMLInputElement>)=>{
    const f=e.target.files?.[0];e.target.value='';if(!f)return;
    try {if(f.size>MAX_JOURNAL_BYTES)throw new Error('Choose a journal smaller than 4 MB.');const imported=parseJournal(await f.text());if(!imported.rounds.length)throw new Error('That journal has no rounds.');setPendingImport(imported);setError('');}
    catch(e){setError(errorText(e));}
  };
  const acceptImport=()=>{
    if(!pendingImport)return;
    // Import as new copies. Existing records and their scenario ancestry remain untouched.
    const ids=new Map(pendingImport.rounds.map(r=>[r.id,cryptoId()]));
    const rounds=pendingImport.rounds.map(r=>({...r,id:ids.get(r.id)!,title:`${r.title.slice(0,65)} · imported`}));
    const importedScenarios=pendingImport.scenarios.map(s=>({...s,id:cryptoId(),parent:{...s.parent,journalId:ids.get(s.parent.journalId)!}}));
    if(persist({...doc,rounds:[...doc.rounds,...rounds],scenarios:[...doc.scenarios,...importedScenarios]})){chooseRound(rounds[0].id);setPendingImport(null);setNotice(`Imported ${rounds.length} round${rounds.length===1?'':'s'} as separate copies`);}
  };
  const recover=()=>{
    try {const raw=localStorage.getItem(JOURNAL_SAVE);if(raw)download(raw,'grenland-journal-backup.json');const fresh={...emptyJournal(),rounds:[createJournalRound(hole,77)]};writeJournal(fresh);setDoc(fresh);setSelectedId(fresh.rounds[0].id);setBlocked(false);setError('');setNotice('Original data exported. New local journal started.');}
    catch(e){setError(errorText(e));}
  };
  const addEvent=(kind:'stroke'|'penalty')=>{
    const s={...newJournalShot(),kind,penaltyStrokes:kind==='penalty'?1:0};
    if(edit({myhraShots:[...record.myhraShots,s]})){chooseShot(s.id);if(kind==='stroke')setPlace('start');}
  };
  const startWhatIf=()=>{
    if(!selected?.start||!onTryShot||!sameCourse)return;
    if(!persist(doc))return;
    onTryShot([...selected.start.point],{journalId:record.id,shotId:selected.id,recordRevision:record.revision,club:selected.club??'7 iron',target:selected.end?[...selected.end.point]:null,kind:'what-if'});onClose();
  };
  let running=0;
  if(cinema&&playable)return <div className="gj-overlay gj-cinema" onKeyDown={e=>e.stopPropagation()}><section className="gj-dialog gj-cinema-controls" ref={dialog} role="dialog" aria-modal="true" aria-label="Illustrative 3D replay" onKeyDown={keyDown}>
    <div className="gj-cinema-heading"><div><span className="gj-eyebrow">REMEMBERED POSITIONS · ILLUSTRATIVE FLIGHT</span><h2>{record.title}</h2><small>Approximate endpoints · synthetic pin · no measured launch or spin</small></div><button aria-label="Close round journal" onClick={onClose}>×</button></div>
    <div className="gj-cinema-row"><label>Shot<select aria-label="Select replay shot" value={selected.id} onChange={e=>chooseShot(e.target.value)}>{record.myhraShots.filter(s=>s.kind==='stroke'&&s.start&&s.end).map((s,i)=><option key={s.id} value={s.id}>{i+1}. {s.club??'Unknown club'}</option>)}</select></label><button onClick={()=>{setCinema(false);setPlaying(false);}}>Back to map</button>{onTryShot&&<button className="gj-primary" onClick={startWhatIf}>Try another shot →</button>}</div>
    <div className="gj-playback"><button onClick={()=>{if(progress>=1)setProgress(0);setPlaying(v=>!v);}} aria-label={playing?'Pause illustrative replay':'Play illustrative replay'}>{playing?'Ⅱ Pause':'▶ Play'}</button><input aria-label="Replay progress" type="range" min={0} max={1000} value={Math.round(progress*1000)} onChange={e=>{setPlaying(false);setProgress(Number(e.target.value)/1000);}}/><output>{Math.round(progress*100)}%</output></div>
  </section></div>;
  return <div className="gj-overlay" onKeyDown={e=>e.stopPropagation()}>
    <section className="gj-dialog" ref={dialog} role="dialog" aria-modal="true" aria-labelledby="gj-title" onKeyDown={keyDown}>
      <header className="gj-header"><div><span className="gj-eyebrow">YOUR GOLF, REMEMBERED</span><h1 id="gj-title">Round journal</h1></div><button className="gj-close" aria-label="Close round journal" onClick={onClose}>×</button></header>
      <div className="gj-toolbar">
        <label className="gj-round-picker">Round<select aria-label="Select journal round" value={record.id} onChange={e=>chooseRound(e.target.value)}>{doc.rounds.map(r=><option key={r.id} value={r.id}>{r.title}</option>)}</select></label>
        <button disabled={blocked||doc.rounds.length>=30} onClick={()=>{const next=createJournalRound(hole);if(persist({...doc,rounds:[...doc.rounds,next]}))chooseRound(next.id);}}>+ New round</button>
        <button onClick={()=>download(JSON.stringify(doc,null,2),'grenland-round-journal.json')}>Export JSON</button>
        <button disabled={blocked} onClick={()=>file.current?.click()}>Import JSON</button><input ref={file} type="file" accept=".json,application/json" hidden onChange={importFile}/>
      </div>
      {error&&<div className="gj-alert" role="alert">{error}{blocked&&<button onClick={recover}>Export backup & start new journal</button>}</div>}
      {pendingImport&&<div className="gj-import"><p>Ready to add <b>{pendingImport.rounds.length} round(s)</b> and {pendingImport.scenarios.length} what-if shots. Current records will be kept.</p><button onClick={acceptImport}>Add imported copies</button><button onClick={()=>setPendingImport(null)}>Cancel import</button></div>}
      <nav className="gj-tabs" aria-label="Journal views">{([['card','Scorecard'],['shots','Myhra replay'],['scenarios',`What-if (${scenarios.length})`]] as const).map(([id,label])=><button key={id} aria-current={tab===id?'page':undefined} onClick={()=>{setTab(id);setPlaying(false);}}>{label}</button>)}</nav>
      <div className="gj-content">
        {tab==='card'&&<>
          <section className="gj-summary"><div><span>{summary.full?'ENTERED SCORECARD':summary.entered?'SCORECARD IN PROGRESS':'ROUND TOTAL ONLY'}</span><strong>{summary.full?summary.total:record.reportedTotal??'—'}<small>strokes</small></strong></div><div><span>{summary.full||summary.entered?'ENTERED HOLES':'HISTORICAL SCORE'}</span><strong>{summary.entered?summary.relative===null?'—':scoreRelative(summary.relative):record.reportedTotal!==null&&record.historicalPar!==null?scoreRelative(record.reportedTotal-record.historicalPar):'—'}</strong><small>{summary.entered?`${summary.entered}/18 holes · ${record.parSource==='preview-assumed'?'preview pars':'entered pars'}`:record.historicalPar===null?'Enter historical par to compare':'vs entered course par'}</small></div><p>{summary.entered===0?'A round total does not tell us the hole scores or shot locations. Add only what you remember; the original total stays separate.':'Gross strokes include penalties. Blank holes remain unknown, and your entered total is kept for comparison.'}</p></section>
          <div className="gj-metadata">
            <label>Round name<input key={`name-${record.id}`} aria-label="Round name" maxLength={80} defaultValue={record.title} onBlur={e=>edit({title:e.target.value||'My Grenland round'})}/></label>
            <label>Date (optional)<input aria-label="Round date" type="date" value={record.date??''} onChange={e=>edit({date:e.target.value||null})}/></label>
            <label>Tee (optional)<input aria-label="Historical tee" maxLength={40} placeholder="Unknown · e.g. 58 / yellow" value={record.tee??''} onChange={e=>edit({tee:e.target.value||null})}/></label>
            <label>Reported total<input key={`total-${record.id}`} aria-label="Reported round total" type="number" inputMode="numeric" min={18} max={720} defaultValue={record.reportedTotal??''} onBlur={e=>edit({reportedTotal:optionalInteger(e.target.value)})}/></label>
            <label>Historical course par<input key={`par-${record.id}`} aria-label="Historical course par" type="number" inputMode="numeric" min={18} max={144} placeholder="Unknown" defaultValue={record.historicalPar??''} onBlur={e=>edit({historicalPar:optionalInteger(e.target.value)})}/></label>
          </div>
          <p className="gj-note">Hole pars initially use the current community preview (72). Edit them to match your historical scorecard. No missing score or shot is generated.</p>
          <div className="gj-table-wrap"><table className="gj-scorecard"><caption>Grenland · 18 holes · manual scorecard</caption><thead><tr><th>Hole</th><th>Par</th><th>Gross</th><th>Putts</th><th>To par</th><th>Running</th></tr></thead><tbody>{record.holes.map((h,i)=>{
            const delta=h.strokes!==null&&h.par!==null?h.strokes-h.par:null;if(delta!==null)running+=delta;const cumulative=record.holes.slice(0,i+1).every(h=>h.strokes!==null&&h.par!==null)?running:null;
            const update=(field:'par'|'strokes'|'putts',value:string)=>edit({holes:record.holes.map((v,n)=>n===i?{...v,[field]:optionalInteger(value)}:v),...(field==='par'?{parSource:'user-entered' as const}:{})});
            return <tr key={h.number}><th scope="row">{h.number===1?'01 · Myhra':String(h.number).padStart(2,'0')}</th><td><input aria-label={`Hole ${h.number} par`} type="number" inputMode="numeric" min={1} max={8} value={h.par??''} onChange={e=>update('par',e.target.value)}/></td><td><input aria-label={`Hole ${h.number} gross score`} type="number" inputMode="numeric" min={1} max={40} placeholder="—" value={h.strokes??''} onChange={e=>update('strokes',e.target.value)}/></td><td><input aria-label={`Hole ${h.number} putts`} type="number" inputMode="numeric" min={0} max={40} placeholder="—" value={h.putts??''} onChange={e=>update('putts',e.target.value)}/></td><td className={delta!==null&&delta<0?'gj-under':''}>{delta===null?'—':scoreRelative(delta)}</td><td>{cumulative===null?'—':scoreRelative(cumulative)}</td></tr>;
          })}</tbody><tfoot><tr><th>Total</th><td>{summary.holePars??'—'}</td><td>{summary.entered?summary.total:'—'}</td><td>—</td><td colSpan={2}>{summary.relative===null?'—':scoreRelative(summary.relative)}</td></tr></tfoot></table></div>
          <button className="gj-primary" onClick={()=>setTab('shots')}>Add Myhra shot positions →</button>
        </>}
        {tab==='shots'&&<>
          <div className="gj-replay-intro"><div><h2>Remember your opening hole</h2><p>Place where each shot started and finished. The connecting animation is illustrative; it is not a measured ball flight.</p></div><span className="gj-badge">MANUAL POSITIONS</span></div>
          {!sameCourse&&<div className="gj-alert">This record belongs to a different course revision. Its original positions are preserved. Editing positions and 3D what-if are disabled; start a new round on the current course.</div>}
          <div className="gj-replay-grid"><section className="gj-map-panel">
            <div className="gj-map-heading"><span>MYHRA · HOLE 01</span><button onClick={()=>setWideMap(v=>!v)}>{wideMap?'Focus on hole':'Wider map'}</button></div>
            <svg ref={map} className={`gj-map ${place?'gj-map-place':''}`} viewBox={wideMap?'0 0 768 768':'250 185 270 420'} preserveAspectRatio="xMidYMid meet" role="application" tabIndex={0} aria-label={place?`Place shot ${place}. Click the map, or use arrow keys and Enter; Shift moves one metre.`:'Myhra shot map. Choose Place start or Place finish to edit a position.'} onKeyDown={mapKey} onPointerUp={e=>{
              if(!place)return;const svg=e.currentTarget,p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const matrix=svg.getScreenCTM();if(!matrix)return;const local=p.matrixTransform(matrix.inverse());placeAt([local.x,0,local.y]);
            }}>
              {mapBackground}
              <path d={`M${record.pin.point[0]} ${record.pin.point[2]+5}v-15l9 3-9 3`} stroke="#fff5d8" strokeWidth={1.5} fill="#dcad55"/><text x={record.pin.point[0]+12} y={record.pin.point[2]-1} fill="#fff5d8" fontSize={8}>Synthetic pin</text>
              {record.myhraShots.map((s,i)=>s.kind==='stroke'?<g key={s.id} opacity={selected?.id===s.id?1:.45}>{s.start&&s.end&&<line x1={s.start.point[0]} y1={s.start.point[2]} x2={s.end.point[0]} y2={s.end.point[2]} stroke={selected?.id===s.id?'#f1cd7e':'#edf2df'} strokeWidth={selected?.id===s.id?2:1} strokeDasharray="4 3"/>}{s.start&&<><circle cx={s.start.point[0]} cy={s.start.point[2]} r={5} fill="#f1cd7e" stroke="#18392e" strokeWidth={1}/><text x={s.start.point[0]} y={s.start.point[2]+2.5} textAnchor="middle" fontSize={7} fill="#15332b">{i+1}</text></>}{s.end&&<circle cx={s.end.point[0]} cy={s.end.point[2]} r={3} fill="#fff"/>}</g>:null)}
              {ball&&<circle cx={ball[0]} cy={ball[2]} r={5.5} fill="#fff" stroke="#13372d" strokeWidth={1.5}/>}
              {place&&<g stroke="#ffde92" strokeWidth={1.5}><circle cx={cursor[0]} cy={cursor[2]} r={8} fill="none"/><path d={`M${cursor[0]-12} ${cursor[2]}h24M${cursor[0]} ${cursor[2]-12}v24`}/></g>}
            </svg>
            <p className="gj-map-instruction" role="status">{place?`Tap the map to place the ${place==='start'?'start':'finish'}. Escape cancels.`:'Positions are manual estimates on an approximate course. The preview pin is not the historical pin.'}</p>
            <div className="gj-playback"><button disabled={!playable} onClick={()=>{setPlace(null);if(progress>=1)setProgress(0);setPlaying(v=>!v);}} aria-label={playing?'Pause illustrative replay':'Play illustrative replay'}>{playing?'Ⅱ Pause':'▶ Play'}</button><input aria-label="Replay progress" type="range" min={0} max={1000} value={Math.round(progress*1000)} disabled={!playable} onChange={e=>{setPlaying(false);setProgress(Number(e.target.value)/1000);}}/><output>{Math.round(progress*100)}%</output></div>
            <small className="gj-note">Illustrative playback · no inferred carry, launch or spin</small>
            {onReplayFrame&&<button disabled={!playable||!sameCourse} onClick={()=>{setPlace(null);setProgress(0);setCinema(true);setPlaying(true);}}>Watch in 3D →</button>}
          </section><section className="gj-shot-editor">
            <div className="gj-event-actions"><button disabled={blocked||record.myhraShots.length>=80} onClick={()=>addEvent('stroke')}>+ Shot</button><button disabled={blocked||record.myhraShots.length>=80} onClick={()=>addEvent('penalty')}>+ Penalty</button></div>
            {!selected?<div className="gj-empty"><h3>No shot positions yet</h3><p>Add your first shot, choose a club, then mark its start and finish. A scorecard alone cannot recreate it.</p></div>:<>
              <label>Selected event<select aria-label="Select replay shot" value={selected.id} onChange={e=>chooseShot(e.target.value)}>{record.myhraShots.map((s,i)=><option key={s.id} value={s.id}>{i+1}. {s.kind==='penalty'?`Penalty +${s.penaltyStrokes}`:s.club??'Unknown club'}{s.kind==='stroke'&&s.penaltyStrokes?` (+${s.penaltyStrokes})`:''}</option>)}</select></label>
              {selected.kind==='stroke'&&<><label>Club<select aria-label="Recorded club" value={selected.club??''} onChange={e=>changeShot({club:e.target.value||null})}><option value="">Unknown</option>{CLUBS.map(c=><option key={c.name}>{c.name}</option>)}</select></label>
                <div className="gj-position"><span>START <small>{selected.start?formatPoint(selected.start.point):'Not recorded'}</small></span><button aria-pressed={place==='start'} disabled={!sameCourse} onClick={()=>{setPlace('start');setPlaying(false);setCursor(selected.start?.point??hole.data.tee);}}>Place start</button></div>
                <div className="gj-position"><span>FINISH <small>{selected.end?formatPoint(selected.end.point):'Not recorded'}</small></span><button aria-pressed={place==='end'} disabled={!sameCourse} onClick={()=>{setPlace('end');setPlaying(false);setCursor(selected.end?.point??hole.data.pin);}}>Place finish</button></div>
                <div className="gj-small-actions"><button disabled={!sameCourse} onClick={()=>changeShot({start:{point:[...hole.data.tee],method:'manual-map',courseRevision:revision}})}>Use preview tee</button><button disabled={!sameCourse||!record.myhraShots.slice(0,record.myhraShots.findIndex(s=>s.id===selected.id)).at(-1)?.end} onClick={()=>{const p=record.myhraShots.slice(0,record.myhraShots.findIndex(s=>s.id===selected.id)).at(-1)?.end;if(p)changeShot({start:structuredClone(p)});}}>Use previous finish</button></div>
              </>}
              <label>{selected.kind==='penalty'?'Penalty strokes':'Penalty strokes after this shot'}<input aria-label="Recorded penalty strokes" type="number" inputMode="numeric" min={selected.kind==='penalty'?1:0} max={10} value={selected.penaltyStrokes} onChange={e=>changeShot({penaltyStrokes:Number(e.target.value)})}/></label>
              <label>Note (optional)<textarea aria-label="Shot note" maxLength={280} rows={2} placeholder="e.g. Dropped beside the pond; position approximate" value={selected.note} onChange={e=>changeShot({note:e.target.value})}/></label>
              <button className="gj-delete" onClick={()=>{if(edit({myhraShots:record.myhraShots.filter(s=>s.id!==selected.id)})){setShotId(null);setPlaying(false);setProgress(0);setPlace(null);}}}>Remove this event</button>
              {selected.kind==='stroke'&&onTryShot&&<button className="gj-primary" disabled={!selected.start||!sameCourse||blocked} onClick={startWhatIf}>Try a different shot in 3D →</button>}
              <p className="gj-note">What-if play creates a separate scenario. Your recorded score and positions stay unchanged.</p>
            </>}
            <div className="gj-event-total"><b>{summary.myhraCount}</b> recorded strokes incl. penalties <span>Scorecard: {record.holes[0].strokes??'not entered'}</span></div>
          </section></div>
        </>}
        {tab==='scenarios'&&<><div className="gj-replay-intro"><div><h2>What might have been</h2><p>Simulation experiments saved separately from your historical round. Distances below are simulated.</p></div><span className="gj-badge">WHAT-IF</span></div>{scenarios.length===0?<div className="gj-empty"><h3>No scenarios yet</h3><p>Place a Myhra shot in the replay tab, then choose “Try a different shot in 3D”.</p><button onClick={()=>setTab('shots')}>Open Myhra replay</button></div>:<div className="gj-scenarios">{scenarios.map((s,i)=><article key={s.id}><span>SCENARIO {i+1} · {new Date(s.createdAt).toLocaleDateString()}</span><h3>{s.club} <small>{Math.round(s.result.distance)} m total</small></h3><p>{Math.round(s.result.carry)} m carry · {s.result.penalty?`${s.result.penalty} penalty stroke(s)`:s.result.made?'Holed':'No penalty'}</p><p className="gj-note">Original record revision {s.parent.recordRevision}. {s.parent.shot.end?`${Math.round(Math.hypot(s.result.end[0]-s.parent.shot.end.point[0],s.result.end[2]-s.parent.shot.end.point[2]))} m from recorded finish.`:'Original finish unknown.'}</p><button onClick={()=>persist({...doc,scenarios:doc.scenarios.filter(v=>v.id!==s.id)})}>Remove scenario</button></article>)}</div>}</>}
        {summary.warnings.length>0&&<aside className="gj-warnings" aria-label="Record discrepancies"><h3>Check your record</h3>{summary.warnings.map((w,i)=><p key={i}>{w}</p>)}</aside>}
      </div>
      <footer className="gj-footer"><span role="status">{notice||'Private to this browser · export to keep a copy'}</span><span>{record.courseRevision} · record {record.revision}</span></footer>
    </section>
  </div>;
}

function cryptoId(){const bytes=new Uint8Array(16);crypto.getRandomValues(bytes);return Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');}
