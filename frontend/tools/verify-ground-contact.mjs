import {chromium} from 'playwright';import {mkdir,writeFile} from 'node:fs/promises';import assert from 'node:assert/strict';
const out='../docs/rebuild/ground-contact';await mkdir(out,{recursive:true});const browser=await chromium.launch({channel:'chrome'}),page=await browser.newPage({viewport:{width:900,height:900}}),checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
async function transforms(){return page.evaluate(()=>{const scene=window.assetScene,feet=[],hands=[];scene.traverse(o=>{if(o.isBone&&/^(mixamorig:?)?(Left|Right)Foot$/.test(o.name)){const p=o.getWorldPosition(o.position.clone());feet.push({name:o.name,position:p.toArray()});}if(o.isBone&&/^(mixamorig:?)?(Left|Right)Hand$/.test(o.name)){const p=o.getWorldPosition(o.position.clone());hands.push({name:o.name,position:p.toArray()});}});return{feet,hands,grip:Array.from(scene.getObjectByName('PlayerClubGrip').matrixWorld.elements)};});}
try{
 for(const [family,club,stance,style] of [['driver',0,.92,'full'],['iron',6,.8,'full'],['chip',11,.8,'chip'],['putt',13,.63,'full']]){
  let flat;
  for(const slope of [0,.3,-.3]){
   await page.goto(`http://127.0.0.1:5175/tools/asset-review.html?asset=golfer&angle=side&club=${club}&style=${style}&slope=${slope}`);await page.waitForFunction(()=>window.assetScene?.getObjectByName('PlayerClubGrip'),null,{timeout:60000});await page.waitForTimeout(400);const a=await transforms();assert.equal(a.feet.length,2);assert.equal(a.hands.length,2);
   const maxFootError=Math.max(...a.feet.map(f=>Math.abs(f.position[1]-.115-slope*(f.position[2]-stance))));assert.ok(maxFootError<.02,`${family} ${slope} foot error ${maxFootError}`);
   await page.waitForTimeout(850);const b=await transforms(),drift=Math.max(...a.feet.flatMap((f,i)=>f.position.map((v,j)=>Math.abs(v-b.feet[i].position[j]))));assert.ok(drift<.0001,'Paused correction must not accumulate');
   if(!slope)flat=a;else {assert.ok(Math.max(...a.grip.map((v,i)=>Math.abs(v-flat.grip[i])))<.0001,'Terrain stance must preserve the authored club');const error=Math.max(...a.hands.flatMap((h,i)=>h.position.map((v,j)=>Math.abs(v-flat.hands[i].position[j]))));assert.ok(error<.02,`Terrain stance must preserve the authored grip: ${family}, slope ${slope}, error ${error}, flat ${JSON.stringify(flat.hands)}, slope ${JSON.stringify(a.hands)}`);}
   await page.screenshot({path:`${out}/${family}-${slope}.png`});checks.push({family,slope,maxFootErrorMetres:maxFootError,pausedDriftMetres:drift,passed:true});
  }console.log('PASS terrain stance',family);
 }assert.deepEqual(errors,[]);
}catch(e){checks.push({passed:false,error:String(e)});console.error(e);process.exitCode=1;}finally{await writeFile(out+'/checks.json',JSON.stringify({kind:'Rendered bone/contact and grip checks on authored ±30% planes. Does not certify shoe mesh deformation or professional technique.',checks,errors},null,2));await browser.close();}
