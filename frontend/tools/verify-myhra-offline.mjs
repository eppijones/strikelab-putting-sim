import {createServer} from 'node:http';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
const root=path.resolve('dist'),legacyHTML='<html><body>Previous release<script>addEventListener("load",()=>navigator.serviceWorker.register("/sw.js"))</script></body></html>';
const legacySW=`self.addEventListener('install',e=>e.waitUntil(caches.open('grenland-test-previous').then(c=>c.put('/index.html',new Response(${JSON.stringify(legacyHTML)},{headers:{'Content-Type':'text/html'}}))).then(()=>self.skipWaiting())));self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));self.addEventListener('fetch',e=>{if(e.request.mode==='navigate')e.respondWith(caches.match('/index.html'));});`;
const mime={'.js':'text/javascript','.html':'text/html','.css':'text/css','.json':'application/json','.webp':'image/webp','.png':'image/png','.hdr':'application/octet-stream','.glb':'model/gltf-binary'};
const server=createServer(async(req,res)=>{try{const pathname=new URL(req.url,'http://localhost').pathname;res.setHeader('Cache-Control','no-cache');if(pathname==='/favicon.ico'){res.writeHead(204);res.end();return;}if(pathname==='/legacy-sw.js'){res.setHeader('Content-Type','text/javascript');res.end(legacySW);return;}if(pathname==='/setup'){res.setHeader('Content-Type','text/html');res.end('<html><body>Setup<script>navigator.serviceWorker.register("/legacy-sw.js",{scope:"/"})</script></body></html>');return;}const file=path.resolve(root,'.'+(pathname.startsWith('/play/')?'/index.html':pathname));if(!file.startsWith(root+path.sep)){res.writeHead(404);res.end();return;}res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.end(await readFile(file));}catch{res.writeHead(404);res.end();}});
await new Promise(resolve=>server.listen(5192,'127.0.0.1',resolve));const browser=await chromium.launch({channel:'chrome',headless:true}),context=await browser.newContext(),page=await context.newPage(),base='http://127.0.0.1:5192';
try{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(base+'/setup');await page.waitForFunction(()=>navigator.serviceWorker.controller!==null);
 console.log('Legacy worker active');await page.goto(base+'/play/grenland/myhra');await page.getByRole('button',{name:'Use this shot',exact:true}).waitFor({timeout:90000});
 assert.ok((await page.evaluate(()=>navigator.serviceWorker.controller.scriptURL)).endsWith('/sw.js'),'New worker must replace the previous cache-first shell');
 console.log('Migrated to new game');await context.setOffline(true);await page.reload();await page.getByRole('button',{name:'Use this shot',exact:true}).waitFor({timeout:90000});
 await page.getByRole('button',{name:'Full course map',exact:true}).click();await page.getByRole('link',{name:'Play hole 2: Mors Ekre',exact:true}).click();await page.waitForSelector('.course-hole-card',{timeout:45000});assert.match(await page.locator('.course-hole-card').innerText(),/Mors Ekre/);
 assert.deepEqual(errors,[]);await writeFile('../docs/myhra-offline-check.json',JSON.stringify({migrationFromPreviousRelease:true,offlineMyhra:true,offlineMapAndHoleNavigation:true,errors},null,2));
 console.log('Previous-release migration and offline Myhra / map / hole navigation passed');
}finally{await browser.close();server.close();}
