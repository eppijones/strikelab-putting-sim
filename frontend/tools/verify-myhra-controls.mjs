import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({channel:'chrome',headless:true}),page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];
page.on('pageerror',e=>errors.push(e.message));
const url=process.env.MYHRA_URL||'http://127.0.0.1:5173/play/grenland/myhra',origin=new URL(url).origin;
const speed=async()=>Number(await page.getByLabel('Cart speed',{exact:true}).textContent());
const ready=()=>page.getByRole('button',{name:'Use this shot',exact:true}).waitFor({timeout:90000});
const aim=()=>page.locator('.mh-aim').innerText();
try{
 await page.addInitScript(()=>{window.testPad=null;Object.defineProperty(navigator,'getGamepads',{value:()=>[window.testPad]});});
 await page.goto(url);await ready();
 // Looking around must never silently change the shot's target bearing.
 const before=await aim();await page.mouse.move(750,430);await page.mouse.down();await page.mouse.move(1030,460,{steps:15});await page.mouse.up();assert.equal(await aim(),before);
 await page.getByRole('button',{name:'Reset shot camera',exact:true}).click();await page.getByRole('button',{name:'Aim right',exact:true}).click();assert.match(await aim(),/RIGHT/);
 await page.keyboard.press('a');assert.match(await aim(),/0.5/,'Keyboard aim works after a UI button receives focus');
 await page.getByRole('button',{name:'Drive the cart',exact:true}).click();
 // Keep an idle standard controller connected while using keyboard/touch.
 await page.evaluate(()=>window.testPad={connected:true,mapping:'standard',axes:[0,0,0,0],buttons:Array.from({length:17},()=>({pressed:false,value:0}))});
 await page.keyboard.down('w');await page.waitForTimeout(850);const normal=await speed();assert.ok(normal>10&&normal<=30,`Normal speed ${normal}`);
 await page.keyboard.down('Shift');await page.waitForTimeout(800);const boosted=await speed();assert.ok(boosted>normal*1.45,`Boost ${boosted}, normal ${normal}`);
 await page.keyboard.up('Shift');await page.keyboard.up('w');await page.waitForTimeout(1500);assert.ok(await speed()<2);
 await page.getByRole('button',{name:'Toggle boost'}).click();assert.equal(await page.getByRole('button',{name:'Toggle boost'}).getAttribute('aria-pressed'),'true');
 const stick=await page.getByLabel('Drive joystick',{exact:true}).boundingBox(),sx=stick.x+stick.width/2,sy=stick.y+stick.height/2;
 await page.mouse.move(sx,sy);await page.mouse.down();await page.mouse.move(sx,sy-38,{steps:5});await page.waitForTimeout(700);assert.ok(await speed()>15,'Circular stick drives with an idle controller connected');await page.mouse.up();await page.waitForTimeout(1600);assert.ok(await speed()<2,'Releasing stick stops the cart');
 await page.getByRole('button',{name:'Back to my ball',exact:true}).click();
 console.log('Drive and aim checks passed');
 // Controller aim / shoulders / analog swing, including actual scoring.
 await page.getByRole('button',{name:'Menu',exact:true}).click();await page.getByRole('button',{name:'Putting · 4 m',exact:false}).click();await ready();await page.getByRole('button',{name:'Use this shot',exact:true}).click();
 const angleValue=text=>Number(text.match(/[\d.]+/)?.[0]??0)*(text.includes('LEFT')?-1:1),padAimBefore=angleValue(await aim());await page.evaluate(()=>window.testPad.axes[0]=.6);await page.waitForTimeout(500);await page.evaluate(()=>window.testPad.axes[0]=0);assert.ok(angleValue(await aim())>padAimBefore+.5,'Right stick direction must increase aim bearing');
 await page.getByRole('button',{name:'Caddie',exact:true}).click();await page.getByRole('button',{name:'Use this shot',exact:true}).click();
 await page.evaluate(()=>window.testPad.axes[3]=.4);await page.waitForTimeout(230);await page.evaluate(()=>window.testPad.axes[3]=1);await page.waitForTimeout(250);
 for(const pull of [.8,.6,.4,.2,0]){await page.evaluate(p=>window.testPad.axes[3]=p,pull);await page.waitForTimeout(65);}
 await page.waitForFunction(()=>JSON.parse(localStorage.getItem('strikelab.myhra.hole.v2')||'null')?.strokes>=1);await page.waitForFunction(()=>!document.querySelector('.mh-playing'),{},{timeout:30000});
 const saved=await page.evaluate(()=>localStorage.getItem('strikelab.myhra.hole.v2'));assert.ok(saved);
 await page.evaluate(()=>window.testPad=null);
 console.log('Controller shot saved');
 // An unfinished preview hole gets a separate save; skipping never invents a score.
 await page.getByRole('button',{name:'Full course map',exact:true}).click();await page.getByRole('dialog',{name:'Full course map',exact:true}).waitFor();await page.screenshot({path:'../docs/myhra-course-map.png'});
 await page.getByRole('link',{name:'Play hole 2: Mors Ekre',exact:true}).click();await page.waitForURL(origin+'/play/grenland?hole=2');await page.locator('.course-preparing').waitFor({state:'hidden',timeout:90000});await page.getByRole('button',{name:'Pull back',exact:false}).count();
 await page.keyboard.press('Escape');await page.locator('canvas').click({position:{x:700,y:430}});
 await page.keyboard.press('Space');await page.waitForTimeout(1250);await page.keyboard.press('Space');await page.waitForTimeout(230);await page.keyboard.press('Space');await page.waitForTimeout(190);await page.keyboard.press('Space');
 await page.waitForFunction(()=>JSON.parse(localStorage.getItem('strikelab.grenland.round.v1.explore.2')||'null')?.strokes>0,{},{timeout:15000});
 console.log('Hole 2 keyboard shot saved');
 const second=await page.evaluate(()=>localStorage.getItem('strikelab.grenland.round.v1.explore.2'));
 await page.waitForFunction(()=>!document.querySelector('.course-status')?.textContent?.includes('Ball in motion'),{},{timeout:25000});
 await page.getByRole('link',{name:'Next hole',exact:true}).click();await page.waitForURL(origin+'/play/grenland?hole=3');await page.getByRole('link',{name:'Previous hole',exact:true}).click();await page.waitForURL(origin+'/play/grenland?hole=2');
 assert.equal(await page.evaluate(()=>localStorage.getItem('strikelab.grenland.round.v1.explore.2')),second);
 await page.getByRole('link',{name:'Previous hole',exact:true}).click();await page.waitForURL(origin+'/play/grenland/myhra');await page.waitForSelector('.mh-score-strip',{timeout:60000});assert.equal(await page.evaluate(()=>localStorage.getItem('strikelab.myhra.hole.v2')),saved);
 assert.deepEqual(errors,[]);await writeFile('../docs/myhra-controls-check.json',JSON.stringify({url,normalKmh:normal,boostKmh:boosted,joystick:true,aimCameraIndependent:true,keyboardSwing:true,standardGamepadSimulated:true,holeNavigationRetainsBothSaves:true,errors},null,2));console.log('Controls, boost, gamepad and hole navigation passed');
}finally{await browser.close();}
