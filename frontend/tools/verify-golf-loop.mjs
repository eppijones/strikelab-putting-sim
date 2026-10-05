import {chromium} from 'playwright';import {writeFile} from 'node:fs/promises';import assert from 'node:assert/strict';
const browser=await chromium.launch({channel:'chrome'}),context=await browser.newContext({viewport:{width:393,height:852},isMobile:true,hasTouch:true}),page=await context.newPage(),cdp=await context.newCDPSession(page),errors=[],shots=[];
page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
const state=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('strikelab.myhra.hole.v3')??'null'));
try{
 await page.goto(process.env.GRENLAND_TEST_URL??'http://127.0.0.1:5175/play/grenland/myhra');
 for(let i=0;i<12;i++){
  await page.waitForFunction(()=>document.querySelector('.mh-swing-pad')?.disabled===false||!!document.querySelector('.mh-result'),null,{timeout:60000});
  if((await state())?.complete)break;
  await page.getByRole('button',{name:'Use this shot',exact:false}).waitFor({timeout:15000});await page.getByRole('button',{name:'Use this shot',exact:false}).click();
  const power=Number(await page.getByLabel('Target power',{exact:true}).inputValue())/100,r=await page.locator('.mh-swing-pad').boundingBox(),x=r.x+r.width*.5,y=r.y+r.height*.20;
  await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y,id:1}]});for(let j=1;j<=8;j++){await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x,y:y+96*power*j/8,id:1}]});await page.waitForTimeout(20);}await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
  await page.waitForFunction(n=>JSON.parse(localStorage.getItem('strikelab.myhra.hole.v3')??'null')?.history.length===n,i+1);
  const round=await state(),h=round.history.at(-1);shots.push({club:h.club,power:h.launch.power,lie:h.lie,end:h.end,penalty:h.penalty,distance:h.distance});console.log('SHOT',i+1,JSON.stringify(shots.at(-1)));
 }
 const complete=await state();assert.equal(complete.complete,true,'caddie-guided tee to cup must complete');await page.locator('.mh-result').waitFor({timeout:60000});await page.screenshot({path:'../docs/rebuild/hole-complete.png'});
 await page.reload();await page.locator('.mh-result').waitFor({timeout:60000});assert.equal((await state()).strokes,complete.strokes);assert.deepEqual(errors,[]);
 await writeFile('../docs/rebuild/golf-loop.json',JSON.stringify({kind:'Chrome with emulated default touch; all shots through UI',passed:true,strokes:complete.strokes,shots,errors},null,2));console.log('PASS complete tee to cup and reload',complete.strokes);
}catch(e){console.error(e);await page.screenshot({path:'../docs/rebuild/golf-loop-failure.png'});await writeFile('../docs/rebuild/golf-loop.json',JSON.stringify({passed:false,shots,errors,error:String(e)},null,2));process.exitCode=1;}finally{await browser.close();}
