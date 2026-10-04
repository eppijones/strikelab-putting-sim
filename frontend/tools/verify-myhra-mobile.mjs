import {webkit} from 'playwright';
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import sharp from 'sharp';
const url=process.env.MYHRA_URL||'http://127.0.0.1:5173/play/grenland/myhra';
const browser=await webkit.launch(),context=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,deviceScaleFactor:2}),page=await context.newPage(),errors=[];
page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
const frameCheck=async()=>{await page.waitForTimeout(1400);const png=await page.screenshot(),meta=await sharp(png).metadata();const stats=await sharp(png).extract({left:Math.floor(meta.width*.60),top:Math.floor(meta.height*.4),width:Math.floor(meta.width*.18),height:Math.floor(meta.height*.15)}).stats();assert.ok(Math.max(...stats.channels.map(c=>c.stdev))>2,'The world must render after changing orientation');};
const saved=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('strikelab.myhra.hole.v2')));
const tap=async locator=>{const b=await locator.boundingBox();assert.ok(b);await page.touchscreen.tap(b.x+b.width/2,b.y+b.height/2);};
try{
 await page.goto(url);await page.getByRole('button',{name:'Use this shot',exact:true}).waitFor({timeout:90000});
 await page.screenshot({path:'../docs/myhra-webkit-phone.png'});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 await tap(page.getByRole('button',{name:'Menu',exact:true}));
 await page.getByLabel('Time of day').fill('17.5');await page.getByLabel('Let the day pass').check();
 await page.getByLabel('Swing controls').selectOption('three-click');
 await tap(page.getByRole('button',{name:'Putting · 4 m',exact:false}));
 await page.getByRole('button',{name:'Use this shot',exact:true}).waitFor({timeout:60000});
 await page.screenshot({path:'../docs/myhra-webkit-green.png'});
 for(let n=0;n<4;n++){
  if(await page.getByLabel('Hole completed').isVisible())break;
  await tap(page.getByRole('button',{name:'Use this shot',exact:true}));
  const pad=page.getByRole('button',{name:'Start timing swing',exact:true});
  await tap(pad);await page.waitForTimeout(1400);await tap(pad);await page.waitForTimeout(220);await tap(pad);await page.waitForTimeout(185);await tap(pad);
  if(n===0){await page.setViewportSize({width:844,height:390});await frameCheck();await page.screenshot({path:'../docs/myhra-webkit-landscape.png'});}
  await page.waitForFunction(()=>document.querySelector('[aria-label="Hole completed"]')||!document.querySelector('.mh-playing'),{},{timeout:30000});
  console.log('mobile shot',n+1,(await saved())?.strokes);
 }
 const before=await saved();assert.ok(before.complete,'Touch timing play should finish the putting exercise');
 await page.reload();await page.getByLabel('Hole completed').waitFor({timeout:60000});assert.equal((await saved()).strokes,before.strokes);
 await page.getByRole('button',{name:'Play Myhra again'}).click();await page.getByRole('button',{name:'Use this shot',exact:true}).waitFor({timeout:60000});
 await page.getByRole('button',{name:'Drive the cart',exact:true}).click();
 await page.setViewportSize({width:390,height:844});await frameCheck();
 await page.screenshot({path:'../docs/myhra-webkit-cart.png'});
 await page.getByRole('button',{name:'Back to my ball',exact:true}).click();
 await page.getByRole('button',{name:'Menu',exact:true}).click();await page.getByRole('button',{name:'Approach · 70 m',exact:false}).click();await page.getByRole('button',{name:'Use this shot',exact:true}).waitFor({timeout:60000});
 await page.screenshot({path:'../docs/myhra-webkit-approach.png'});
 assert.equal(errors.length,0,errors.join('\n'));
 await writeFile('../docs/myhra-mobile-check.json',JSON.stringify({url,engine:'WebKit',touchTimingCompleted:before.complete,strokes:before.strokes,resume:true,rotation:true,dayControl:true,errors},null,2));
 console.log('WebKit touch, rotation, save/resume, cart and day controls passed');
}finally{await browser.close();}
