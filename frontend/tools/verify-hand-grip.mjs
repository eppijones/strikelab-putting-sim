import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const out='../docs/rebuild/grip';await mkdir(out,{recursive:true});
const browser=await chromium.launch({channel:'chrome'}),page=await browser.newPage({viewport:{width:900,height:900}}),checks=[];
try{
 for(const quality of ["high","mobile"])for(const [family,club,style] of [['driver',0,'full'],['iron',6,'full'],['chip',11,'chip'],['putt',13,'full']]){
  await page.goto(`http://127.0.0.1:5175/tools/asset-review.html?asset=golfer&club=${club}&style=${style}&quality=${quality}`);await page.waitForFunction(()=>window.assetScene?.getObjectByName('PlayerClubGrip'));await page.waitForTimeout(350);
  for(const power of [1,.45,.08]){
   let maxRadialGap=0,maxAcrossDot=-1;
   for(const [phase,count] of [['back',10],['down',12],['follow',8]])for(let i=0;i<=count;i++){
    await page.evaluate(({phase,i,count,power})=>{const s=window.reviewSwing.current;s.phase=phase==='back'?'backswing':'finish';s.amount=power*(phase==='back'?i/count:1);s.peak=s.amount;s.lockedPower=power;window.reviewTime.current=phase==='back'?-1:phase==='down'?i/count*.24:.24+i/count*.8;},{phase,i,count,power});await page.waitForTimeout(35);
    const sample=await page.evaluate(()=>{
     const scene=window.assetScene,bones={};scene.traverse(o=>{if(o.isBone)bones[o.name.replace('mixamorig','').replace(':','')]=o;});const m=scene.getObjectByName('PlayerClubGrip').matrixWorld.elements,origin=[m[12],m[13],m[14]],dir=[-m[4],-m[5],-m[6]],dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),pos=n=>bones[n].getWorldPosition(bones[n].position.clone()).toArray();
     return ['Left','Right'].map(side=>{const centre=[0,0,0];for(const suffix of ['Middle1','Middle3','Middle4','Ring1','Ring3','Ring4'])pos(side+'Hand'+suffix).forEach((v,i)=>centre[i]+=v/6);const delta=centre.map((v,i)=>v-origin[i]),along=dot(delta,dir),across=pos(side+'HandIndex1').map((v,i)=>v-pos(side+'HandPinky1')[i]),length=Math.hypot(...across);return {side,radial:Math.sqrt(Math.max(0,dot(delta,delta)-along*along)),along,acrossDot:dot(across,dir)/length};});
    });
    for(const hand of sample){maxRadialGap=Math.max(maxRadialGap,hand.radial);maxAcrossDot=Math.max(maxAcrossDot,hand.acrossDot);assert.ok(hand.radial<.025,`${family} ${power} ${phase} ${i}: ${hand.side} grip gap ${hand.radial}`);assert.ok(hand.acrossDot<-.65,`${family} ${power}: ${hand.side} index/pinky orientation inverted ${hand.acrossDot}`);}
   }
   checks.push({quality,family,power,maxRadialGapMetres:maxRadialGap,maxAcrossDot,passed:true});
  }console.log('PASS animated grip',family);
 }
}catch(error){checks.push({passed:false,error:String(error)});console.error(error);process.exitCode=1;}
await writeFile(out+'/checks.json',JSON.stringify({kind:'Rendered authored finger-centre/shaft and anatomical index/pinky orientation checks throughout captured takeaway, release and follow-through. 25mm centre tolerance, not coach certification.',checks},null,2));await browser.close();
