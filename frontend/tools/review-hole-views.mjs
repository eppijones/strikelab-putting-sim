import {chromium} from 'playwright';
import {readFileSync} from 'node:fs';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {BALL_RADIUS_M,bearingTo,heightAt,surfaceAt} from '../src/course/engine.ts';
import {freshHole} from '../src/myhra/round.ts';

// Deliberately authored QA positions in an isolated browser. Never historical shots.
const root=new URL('../public/courses/grenland/',import.meta.url);
const data=JSON.parse(readFileSync(new URL('myhra/hole.json',root)));
function tile(s){const h=readFileSync(new URL(s.url,root)),l=readFileSync(new URL(s.surfaces,root));return {...s,heights:new Float32Array(h.buffer.slice(h.byteOffset,h.byteOffset+h.byteLength)),lies:new Uint8Array(l)};}
const world={terrain:tile(data.terrain),detail:tile(data.detail),pin:data.pin,hazards:data.regions.filter(r=>r.kind==='water'),stimp:10,wind:[0,0]};
for(const p of [data.pin,data.tee])p[1]=heightAt(world,p[0],p[2]);
const hole={world,data},out='../docs/rebuild/views';await mkdir(out,{recursive:true});
const url=process.env.GRENLAND_TEST_URL??'http://127.0.0.1:4175/play/grenland/myhra';
const browser=await chromium.launch({channel:'chrome'}),context=await browser.newContext({viewport:{width:1280,height:720},hasTouch:true,recordVideo:{dir:out,size:{width:1280,height:720}}});
const page=await context.newPage(),cdp=await context.newCDPSession(page),errors=[],checks=[];
page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
const ready=()=>page.waitForFunction(()=>document.querySelector('.mh-swing-pad')?.disabled===false,null,{timeout:60000});
const state=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('strikelab.myhra.hole.v3')));
async function position(name,x,z){const ball=[x,heightAt(world,x,z),z],round={...freshHole(hole),ball};await page.evaluate(r=>localStorage.setItem('strikelab.myhra.hole.v3',JSON.stringify(r)),round);await page.reload();await ready();await page.waitForTimeout(1400);await page.screenshot({path:out+'/'+name+'.png'});checks.push({view:name,position:ball,lie:surfaceAt(world,x,z),provenance:'authored QA fixture'});return ball;}
async function stroke(power){const r=await page.locator('.mh-swing-pad').boundingBox(),x=r.x+r.width*.5,y=r.y+r.height*.2;await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y,id:1}]});for(let i=1;i<=8;i++){await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x,y:y+96*power*i/8,id:1}]});await page.waitForTimeout(24);}await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});}
async function view(name,button){await page.getByRole('button',{name:button,exact:true}).click();await page.waitForTimeout(1800);await page.screenshot({path:out+'/'+name+'.png'});}
try{
 await page.goto(url);await ready();await page.waitForTimeout(1800);await page.screenshot({path:out+'/tee.png'});
 await view('tee-overview','Scout the hole');await view('tee-return','Return to golfer');
 await position('approach',355,374);
 const bunker=await position('bunker',365.67,343.50);assert.equal(surfaceAt(world,bunker[0],bunker[2]),'bunker');
 await page.getByRole('button',{name:'Choose club',exact:true}).click();await page.getByRole('button',{name:/^SW /}).click();await stroke(.7);await ready();const bunkerShot=(await state()).history[0];assert.equal(bunkerShot.lie,'bunker');assert.equal(bunkerShot.penalty,0);checks.push({name:'Bunker stroke plays and saves through UI',passed:true,distance:bunkerShot.distance,end:bunkerShot.end});
 const waterBall=await position('greenside-water',414,269);await view('water-overview','Scout the hole');await view('water-return','Return to golfer');
 await page.getByRole('button',{name:'Choose club',exact:true}).click();await page.getByRole('button',{name:/^PW /}).click();
 const steps=Math.round(-bearingTo(waterBall,data.pin)*180/Math.PI/.5);for(let i=0;i<Math.abs(steps);i++)await page.keyboard.press(steps>0?'ArrowRight':'ArrowLeft');
 await stroke(.5);await ready();const relief=await state();assert.equal(relief.strokes,2);assert.equal(relief.history.length,1);assert.equal(relief.history[0].penalty,1);assert.equal(relief.ball[0],waterBall[0]);assert.equal(relief.ball[2],waterBall[2]);assert.ok(Math.abs(relief.ball[1]-waterBall[1]-BALL_RADIUS_M)<1e-6);await page.reload();await ready();assert.equal((await state()).strokes,2);checks.push({name:'Water relief counts shot and penalty once, preserves ground position on reload',passed:true});
 await position('putting',data.pin[0],data.pin[2]+4);await view('putting-read','Read the green');await view('putting-address','Reset shot camera');
 for(const hour of [7,12,18]){await page.getByRole('button',{name:'Menu',exact:true}).click();await page.getByLabel('Time of day',{exact:true}).fill(String(hour));await page.getByRole('button',{name:'Close dialog',exact:true}).click();await page.waitForTimeout(1200);await page.screenshot({path:out+'/daylight-'+hour+'.png'});}
 assert.deepEqual(errors,[]);checks.push({name:'No runtime errors during view transitions',passed:true});
 await writeFile(out+'/review.json',JSON.stringify({url,kind:'Authored QA fixtures, desktop Chrome, screenshots and moving-camera footage. Not surveyed course evidence or physical-device acceptance.',passed:true,checks,errors},null,2));console.log('PASS',JSON.stringify(checks));
}catch(e){console.error(e);await page.screenshot({path:out+'/failure.png'});await writeFile(out+'/review.json',JSON.stringify({passed:false,checks,errors,error:String(e)},null,2));process.exitCode=1;}
finally{await context.close();await page.video().saveAs(out+'/hole-views.webm');await browser.close();}
