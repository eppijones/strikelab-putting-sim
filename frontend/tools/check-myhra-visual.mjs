import {chromium} from 'playwright';
const browser=await chromium.launch({channel:'chrome',headless:true});
for(const mobile of [false,true]){
 const context=await browser.newContext({viewport:mobile?{width:390,height:844}:{width:1440,height:900},isMobile:mobile,hasTouch:mobile,deviceScaleFactor:1});
 const page=await context.newPage();page.on('pageerror',e=>console.error(e));await page.goto('http://127.0.0.1:5173/play/grenland/myhra');await page.getByRole('button',{name:'Use this shot'}).waitFor({state:'visible',timeout:60000});await page.waitForTimeout(4000);await page.screenshot({path:'../docs/myhra-'+(mobile?'phone':'forest')+'.png'});
 if(!mobile){await page.getByRole('button',{name:'Scout the hole'}).click();await page.waitForTimeout(500);await page.screenshot({path:'../docs/myhra-overview.png'});}
 await context.close();
}
await browser.close();
