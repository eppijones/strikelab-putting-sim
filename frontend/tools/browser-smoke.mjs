import {chromium,webkit,devices} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const engine=process.argv[2]??'chromium',url=process.argv[3]??'http://127.0.0.1:5173/play/grenland';
const out=new URL('../../docs/',import.meta.url);await mkdir(out,{recursive:true});
const browser=await (engine==='webkit'?webkit:chromium).launch(engine==='webkit'?{headless:true}:{headless:true,channel:'chrome'});
const context=await browser.newContext({...devices['iPhone 13'],deviceScaleFactor:1});
const page=await context.newPage(),errors=[],requests=[],checks=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('response',r=>{if(r.url().startsWith(new URL(url).origin)&&r.status()>=400)requests.push({url:r.url(),status:r.status()});});
const shot=page.getByRole('button',{name:'Click swing timing',exact:true});
const save=async name=>{await page.waitForTimeout(450);return page.screenshot({path:new URL(`grenland-${engine}-${name}.png`,out).pathname.replace(/^\/([A-Z]:)/,'$1')});};
try{
 await page.goto(url,{waitUntil:'networkidle'});
 await page.getByRole('button',{name:'Drag down then up to swing',exact:true}).waitFor();
 await page.waitForFunction(()=>document.querySelector('canvas')?.width>0&&!document.querySelector('.swing-pad')?.disabled);
 await save('phone');checks.push('Course loaded on phone viewport');
 await page.getByRole('button',{name:'Settings',exact:true}).tap();
 await page.getByRole('combobox',{name:'Swing controls',exact:true}).selectOption('three-click');
 await page.getByRole('button',{name:'4 m putt',exact:true}).tap();
 await page.getByRole('button',{name:'Caddie suggestion',exact:true}).tap();
 await page.waitForFunction(()=>document.querySelector('.course-caddie-note')?.textContent.includes('promising'));
 // Timed keyboard events target the rendered control, exercising the entire UI → swing → physics → save path.
 await shot.focus();
 await page.evaluate(async()=>{const el=document.querySelector('.swing-pad');const press=()=>el.dispatchEvent(new KeyboardEvent('keydown',{code:'Space',key:' ',bubbles:true}));press();await new Promise(r=>setTimeout(r,1500));press();await new Promise(r=>setTimeout(r,238));press();await new Promise(r=>setTimeout(r,200));press();});
 await page.getByRole('button',{name:'Walk to next tee',exact:true}).waitFor({timeout:20000});
 assert.match(await page.locator('.course-round-badge').innerText(),/1 THRU/);checks.push('Timed putt scored and completed hole');await save('cup');
 await page.reload({waitUntil:'networkidle'});
 assert.match(await page.locator('.course-round-badge').innerText(),/1 THRU/);checks.push('Score survived reload');
 await page.getByRole('button',{name:'Walk to next tee',exact:true}).tap();
 assert.match(await page.locator('h1').innerText(),/02/);
 await page.getByRole('button',{name:'Return to ball',exact:true}).waitFor();
 await page.keyboard.down('w');await page.waitForTimeout(1000);await page.keyboard.up('w');await save('walking');checks.push('Walking transition to hole 2');
 await page.getByRole('button',{name:'Drive cart',exact:true}).tap();
 await page.keyboard.down('w');await page.waitForTimeout(1000);await page.keyboard.up('w');await save('cart');checks.push('Cart mode and movement');
 await page.getByRole('button',{name:'Return to ball',exact:true}).tap();
 await page.getByRole('button',{name:'Scout view · V',exact:true}).tap();await save('scout');checks.push('Returned to ball and scouted shot');
 await page.setViewportSize({width:844,height:390});await page.waitForTimeout(1000);await save('landscape');
 // A solid missing canvas compresses to a few hundred bytes. This clear patch
 // of ground must retain texture after rotation, not only working HTML controls.
 const ground=await page.screenshot({clip:{x:304,y:164,width:100,height:40}});
 assert.ok(ground.length>1000,'The course disappeared after orientation change');checks.push('Course remained rendered after orientation change');
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);assert.equal(overflow,false);checks.push('Landscape layout without horizontal overflow');
 assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
}finally{
 await writeFile(new URL(`grenland-${engine}-verification.json`,out),JSON.stringify({engine,url,device:'iPhone 13 emulation; no physical hardware',checks,errors,requests,date:new Date().toISOString()},null,2));
 await browser.close();
}
console.log(JSON.stringify({engine,checks,errors,requests}));
