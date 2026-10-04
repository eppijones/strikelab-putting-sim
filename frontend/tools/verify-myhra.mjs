import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
const root=new URL('../../docs/',import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1');
await mkdir(root,{recursive:true});
const browser=await chromium.launch({channel:'chrome',headless:true}),page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];
page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
const url=process.env.MYHRA_URL||'http://127.0.0.1:5173/play/grenland/myhra';
await page.goto(url);await page.getByRole('button',{name:'Use this shot',exact:true}).waitFor({timeout:60000});
await page.screenshot({path:root+'myhra-desktop.png'});
for(let stroke=0;stroke<9;stroke++){
 if(await page.getByLabel('Hole completed').isVisible())break;
 const apply=page.getByRole('button',{name:'Use this shot',exact:true});await apply.waitFor({timeout:45000});await apply.click();
 const pad=page.getByRole('button',{name:'Pull back and swing through',exact:true});await pad.waitFor();
 const b=await pad.boundingBox(),x=b.x+b.width/2,y=b.y+b.height/2,range=Math.max(30,Math.min(110,900-y-10));
 await page.mouse.move(x,y);await page.mouse.down();
 for(let i=1;i<=10;i++){await page.mouse.move(x,y+range*i/10);await page.waitForTimeout(50);}
 if(stroke===0)await page.screenshot({path:root+'myhra-backswing.png'});
 for(let i=9;i>=0;i--){await page.mouse.move(x,y+range*i/10);await page.waitForTimeout(29);}
 await page.mouse.up();await page.waitForTimeout(350);if(stroke===0)await page.screenshot({path:root+'myhra-followthrough.png'});
 await page.waitForFunction(()=>document.querySelector('[aria-label="Hole completed"]')||!document.querySelector('.myhra-game')?.classList.contains('mh-playing'),{},{timeout:35000});
 console.log('stroke',stroke+1,await page.locator('.mh-score-strip').innerText());
}
const completed=await page.getByLabel('Hole completed').isVisible();
await page.screenshot({path:root+'myhra-round.png'});
await page.getByRole('button',{name:'Drive the cart',exact:true}).click();await page.waitForTimeout(1200);await page.screenshot({path:root+'myhra-cart.png'});
await page.locator('canvas').click({position:{x:720,y:400}});await page.keyboard.down('w');await page.waitForTimeout(3500);await page.keyboard.up('w');await page.screenshot({path:root+'myhra-driving.png'});
await page.getByRole('button',{name:'Back to my ball',exact:true}).click();
await writeFile(root+'myhra-browser-check.json',JSON.stringify({url,completed,errors,save:await page.evaluate(()=>JSON.parse(localStorage.getItem('strikelab.myhra.hole.v2')))},null,2));
console.log(JSON.stringify({completed,errors}));await browser.close();if(!completed||errors.length)process.exitCode=1;
