import {chromium} from 'playwright';import {mkdir,writeFile,copyFile} from 'node:fs/promises';import assert from 'node:assert/strict';
const out='../docs/rebuild/impact';await mkdir(out,{recursive:true});const browser=await chromium.launch({channel:'chrome'}),context=await browser.newContext({viewport:{width:900,height:900},recordVideo:{dir:out,size:{width:900,height:900}}}),page=await context.newPage(),checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 for(const [family,club,style] of [['driver',0,'full'],['iron',6,'full'],['chip',11,'chip'],['putt',13,'full']]){
  await page.goto(`http://127.0.0.1:5175/tools/asset-review.html?asset=golfer&angle=side&club=${club}&style=${style}`);await page.waitForFunction(()=>window.assetScene?.getObjectByName('PlayerClubGrip'),null,{timeout:60000});await page.waitForTimeout(500);let full;
  for(const power of [1,.45,.08]){
   await page.evaluate(power=>{window.reviewTime.current=-1;window.reviewSwing.current.phase='backswing';window.reviewSwing.current.amount=power;window.reviewSwing.current.peak=power;window.reviewSwing.current.lockedPower=power;},power);await page.waitForTimeout(150);
   await page.evaluate(()=>{window.reviewSwing.current.phase='finish';window.reviewTime.current=.24;});await page.waitForTimeout(80);const matrix=await page.evaluate(()=>Array.from(window.assetScene.getObjectByName('PlayerClubGrip').matrixWorld.elements));assert.ok(matrix.every(Number.isFinite));if(!full)full=matrix;else assert.ok(Math.max(...matrix.map((x,i)=>Math.abs(x-full[i])))<1e-6,'Partial power must reach the full square impact transform');
   await page.screenshot({path:`${out}/${family}-${power}-impact.png`});checks.push({family,power,passed:true,maxImpactTransformDifference:Math.max(...matrix.map((x,i)=>Math.abs(x-full[i])))});
   // Capture an actual short stroke through address, partial top, impact and follow.
   for(let i=0;i<=12;i++){await page.evaluate(({i,power})=>{window.reviewTime.current=-1;window.reviewSwing.current.phase='backswing';window.reviewSwing.current.amount=power*i/12;},{i,power});await page.waitForTimeout(25);}
   await page.evaluate(power=>{window.reviewSwing.current.phase='finish';window.reviewSwing.current.amount=power;},power);for(let i=0;i<=24;i++){await page.evaluate(i=>window.reviewTime.current=i/24*1.1,i);await page.waitForTimeout(30);}
  }console.log('PASS impact',family);
 }assert.deepEqual(errors,[]);
}catch(error){checks.push({passed:false,error:String(error)});console.error(error);process.exitCode=1;}
await page.close();const video=await page.video().path();await context.close();await copyFile(video,out+'/partial-strokes.webm');await writeFile(out+'/checks.json',JSON.stringify({kind:'Rendered authored grip transform at 0.24 s release impact, 1.24 s clip time; deterministic full/45%/8% strength. Not coach certification.',checks,errors},null,2));await browser.close();
