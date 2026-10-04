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
self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k.startsWith('grenland-')).slice(0,-2).filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',e=>{const u=new URL(e.request.url);if(e.request.method!=='GET'||u.origin!==self.location.origin||u.pathname.startsWith('/api/')||u.pathname.startsWith('/ws'))return;const key=e.request.mode==='navigate'?'/index.html':u.pathname;e.respondWith(caches.open(CACHE).then(async c=>(await c.match(key))||(await caches.match(key))||fetch(e.request)));});`);
console.log(`Offline cache ${version}: ${urls.length} files`);
