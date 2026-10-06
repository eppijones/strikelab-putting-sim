import {chromium} from 'playwright';import {writeFile} from 'node:fs/promises';import assert from 'node:assert/strict';
const url=process.env.GRENLAND_TEST_URL??'http://127.0.0.1:4175/play/grenland/myhra',b=await chromium.launch({channel:'chrome'}),p=await b.newPage({viewport:{width:393,height:852},isMobile:true,hasTouch:true}),samples=[];
try{
 await p.goto(url);await p.waitForFunction(()=>document.querySelector('.mh-swing-pad')?.disabled===false,null,{timeout:60000});await p.getByRole('button',{name:'Menu',exact:true}).click();await p.getByRole('button',{name:/Putting · 4 m/}).click();await p.waitForFunction(()=>document.querySelector('.mh-swing-pad')?.disabled===false);
 for(let i=0;i<20;i++){
  const value=i%2?70:35;
  await p.evaluate(value=>{
   const input=document.querySelector('input[aria-label="Target power"]'),carry=document.querySelector('.mh-carry b');window.qaLatency=null;
   const previous=carry.textContent;input.addEventListener('input',()=>{const start=performance.now();let response=null,last='',stable=0;const tick=()=>{const now=performance.now(),shown=document.querySelector('.mh-shot-adjust label b').textContent,projected=carry.textContent;if(response===null&&shown===value+'%')response=now-start;if(projected!==previous){stable=projected===last?stable+1:0;last=projected;if(stable>=2){window.qaLatency={responseMs:response,predictionSettledMs:now-start,value,projected};return;}}if(now-start>1000){window.qaLatency={timeout:true,value};return;}requestAnimationFrame(tick);};requestAnimationFrame(tick);},{once:true});
  },value);
  await p.getByLabel('Target power',{exact:true}).fill(String(value));await p.waitForFunction(()=>window.qaLatency!==null);const sample=await p.evaluate(()=>window.qaLatency);assert.ok(!sample.timeout);samples.push(sample);
 }
 const response=samples.map(s=>s.responseMs).sort((a,b)=>a-b),prediction=samples.map(s=>s.predictionSettledMs).sort((a,b)=>a-b),at=(a,q)=>a[Math.floor((a.length-1)*q)],result={kind:'Chrome headless DOM-input event to next rendered power value and stable projected-distance DOM. Does not measure touch-glass, Bluetooth, display scanout or physical-device latency.',url,responseP95Ms:at(response,.95),responseMaxMs:at(response,1),predictionP95Ms:at(prediction,.95),predictionMaxMs:at(prediction,1),samples};assert.ok(result.responseMaxMs<=75);assert.ok(result.predictionMaxMs<=100);console.log(JSON.stringify(result));await writeFile('../docs/rebuild/input-latency.json',JSON.stringify(result,null,2));
}finally{await b.close();}
