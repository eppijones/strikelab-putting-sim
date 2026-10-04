import {readdir,readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
const root=path.resolve('dist');
async function walk(dir){return (await Promise.all((await readdir(dir,{withFileTypes:true})).map(async e=>e.isDirectory()?walk(path.join(dir,e.name)):[path.join(dir,e.name)]))).flat();}
const files=(await walk(root)).filter(f=>!f.endsWith('sw.js')).sort();
const hash=createHash('sha256');for(const file of files)hash.update(await readFile(file));
const version=hash.digest('hex').slice(0,16),urls=files.map(f=>'/'+path.relative(root,f).split(path.sep).join('/'));
await writeFile(path.join(root,'sw.js'),`const CACHE='grenland-${version}';const URLS=${JSON.stringify(urls)};
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(URLS)).then(()=>self.skipWaiting())));
self.addEventListener('activate',e=>e.waitUntil((async()=>{
 const keys=(await caches.keys()).filter(k=>k.startsWith('grenland-')),previous=keys.filter(k=>k!==CACHE);
 const knowsMyhra=(await Promise.all(previous.map(async k=>(await caches.open(k)).match('/courses/grenland/myhra/hole.json')))).some(Boolean);
 await Promise.all(previous.slice(0,-2).map(k=>caches.delete(k)));await self.clients.claim();
 // One-time migration: the old cache-first shell did not know this new route.
 // A round is saved before its shot animation, so this preserves its score.
 // Do not await navigate inside activation: its fetch waits for activation.
 if(previous.length&&!knowsMyhra)for(const client of await self.clients.matchAll({type:'window'}))if(new URL(client.url).pathname.startsWith('/play/grenland/myhra'))client.navigate(client.url).catch(()=>{});
})()));
self.addEventListener('fetch',e=>{const u=new URL(e.request.url);if(e.request.method!=='GET'||u.origin!==self.location.origin||u.pathname.startsWith('/api/')||u.pathname.startsWith('/ws'))return;
 e.respondWith((async()=>{const c=await caches.open(CACHE);
  if(e.request.mode==='navigate'){try{const response=await fetch(e.request,{signal:AbortSignal.timeout(4000)});if(response.ok)return response;}catch{}return (await c.match('/index.html'))||(await caches.match('/index.html'))||Response.error();}
  return (await c.match(u.pathname))||(await caches.match(u.pathname))||fetch(e.request);
 })());});`);
console.log(`Offline cache ${version}: ${urls.length} files`);
