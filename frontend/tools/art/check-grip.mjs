import {chromium} from 'playwright';
const browser=await chromium.launch({channel:'chrome'});const p=await browser.newPage({viewport:{width:1000,height:900}});
await p.goto('http://127.0.0.1:5175/tools/asset-review.html?asset=golfer&close=1&angle=front');await p.waitForTimeout(2800);await p.screenshot({path:'../docs/rebuild/grip-front-corrected.png'});
await p.goto('http://127.0.0.1:5175/tools/asset-review.html?asset=golfer&close=1&angle=side');await p.waitForTimeout(1800);await p.screenshot({path:'../docs/rebuild/grip-side-corrected.png'});await browser.close();
