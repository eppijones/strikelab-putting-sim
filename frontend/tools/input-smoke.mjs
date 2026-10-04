import {chromium,devices} from 'playwright';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const url=process.argv[2]??'http://127.0.0.1:8088/play/grenland';
const browser=await chromium.launch({headless:true,channel:'chrome'});
const context=await browser.newContext({...devices['iPhone 13'],deviceScaleFactor:1});
await context.addInitScript(()=>{
 window.inputTestPad={id:'Standard gamepad test double',connected:true,mapping:'standard',index:0,axes:[0,0,0,0],buttons:Array.from({length:17},()=>({pressed:false,value:0,touched:false}))};
 Object.defineProperty(navigator,'getGamepads',{value:()=>[window.inputTestPad]});
});
const page=await context.newPage(),errors=[],checks=[];page.on('pageerror',e=>errors.push(e.message));
const pad=page.getByRole('button',{name:'Drag down then up to swing',exact:true});
const strokes=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('strikelab.grenland.round.v1.preview')).strokes);
try{
 await page.goto(url,{waitUntil:'networkidle'});await page.waitForFunction(()=>document.querySelector('.swing-pad')&&!document.querySelector('.swing-pad').disabled);
 await page.evaluate(async()=>{
  window.inputTestPad.axes[3]=1;await new Promise(r=>setTimeout(r,650));
  const start=performance.now();await new Promise(resolve=>{const tick=now=>{const progress=Math.min(1,(now-start)/340);window.inputTestPad.axes[3]=1-progress;if(progress<1)requestAnimationFrame(tick);else resolve();};requestAnimationFrame(tick);});
 });
 await page.waitForFunction(()=>document.querySelector('.course-strike-feedback')?.textContent.includes('100% swing'),{},{timeout:20000});
 assert.equal(await strokes(),1);checks.push('Mocked right-stick swing delivered exactly one full-power shot');
 const box=await pad.boundingBox(),x=box.x+box.width/2,y=box.y+box.height/2,range=Math.max(30,Math.min(100,844-y-12));
 await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x,y+range,{steps:20});await page.waitForTimeout(100);
 assert.ok(Number(await page.getByRole('meter',{name:'Swing power'}).getAttribute('aria-valuenow'))>=95);assert.equal(await strokes(),1);
 checks.push('Idle connected gamepad did not interrupt mouse backswing');
 await page.mouse.move(x,y,{steps:20});await page.mouse.up();
 await page.waitForFunction(()=>JSON.parse(localStorage.getItem('strikelab.grenland.round.v1.preview')).strokes===2);
 await page.waitForFunction(()=>!document.querySelector('.swing-pad').disabled,{},{timeout:20000});
 assert.equal(await strokes(),2);checks.push('Captured mobile-size mouse gesture hit once with room below the control');
 if(new URL(url).port!=='5173'){
  await page.evaluate(()=>navigator.serviceWorker.ready);await context.setOffline(true);await page.reload({waitUntil:'load'});
  await page.waitForFunction(()=>document.querySelector('canvas')?.width>0&&!document.querySelector('.swing-pad')?.disabled);
  assert.equal(await strokes(),2);checks.push('Offline reload restored the game and saved round');await context.setOffline(false);
 }
 assert.deepEqual(errors,[]);
}finally{
 await writeFile(new URL('../../docs/grenland-input-verification.json',import.meta.url),JSON.stringify({url,checks,errors,device:'Emulation and mocked gamepad; not physical DualSense validation',date:new Date().toISOString()},null,2));await browser.close();
}
console.log(JSON.stringify({checks,errors}));
