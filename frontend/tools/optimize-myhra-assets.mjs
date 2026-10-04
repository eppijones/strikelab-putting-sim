import sharp from 'sharp';
import {readFile,writeFile,mkdir,copyFile} from 'node:fs/promises';
import path from 'node:path';

const raw=process.argv[2],out=path.resolve('public/courses/grenland/myhra');
await mkdir(out,{recursive:true});
const source=JSON.parse(await readFile(path.join(raw,'fir/fir.gltf'),'utf8'));
// Select the first source tree and repack its referenced buffer ranges before
// parsing. The original contains several million triangles across three trees.
const mesh=source.meshes[0],accessorIds=new Set();
for(const p of mesh.primitives){accessorIds.add(p.indices);Object.values(p.attributes).forEach(i=>accessorIds.add(i));}
const accessorMap=new Map([...accessorIds].map((v,i)=>[v,i]));
const viewIds=[...new Set([...accessorIds].map(i=>source.accessors[i].bufferView))];
const viewMap=new Map(viewIds.map((v,i)=>[v,i]));
const oldBuffer=await readFile(path.join(raw,'fir/fir_tree_01.bin'));
let length=0;const pieces=[];
const views=viewIds.map(id=>{const old=source.bufferViews[id];const bytes=oldBuffer.subarray(old.byteOffset??0,(old.byteOffset??0)+old.byteLength);const view={...old,buffer:0,byteOffset:length};pieces.push(bytes);length+=bytes.length;const padding=(4-length%4)%4;if(padding){pieces.push(Buffer.alloc(padding));length+=padding;}return view;});
source.accessors=[...accessorIds].map(i=>({...source.accessors[i],bufferView:viewMap.get(source.accessors[i].bufferView)}));
for(const p of mesh.primitives){p.indices=accessorMap.get(p.indices);for(const key of Object.keys(p.attributes))p.attributes[key]=accessorMap.get(p.attributes[key]);}
source.meshes=[mesh];source.nodes=[{mesh:0,name:'Scots fir'}];source.scenes=[{nodes:[0]}];source.scene=0;source.bufferViews=views;source.buffers=[{uri:'selected.bin',byteLength:length}];
await writeFile(path.join(raw,'fir/selected.bin'),Buffer.concat(pieces));
await writeFile(path.join(raw,'fir/selected.gltf'),JSON.stringify(source));
await copyFile(path.join(raw,'morning.hdr'),path.join(out,'morning.hdr'));
for(const name of ['grass_ground','gravel_road','sand_02'])for(const kind of ['diff','nor_gl']){
 await sharp(path.join(raw,'materials',name+'-'+kind+'.jpg')).resize(1024,1024).webp({quality:92}).toFile(path.join(out,name+'-'+kind+'.webp'));
}
console.log('Prepared source tree, HDR environment and 1K photographic materials. Run bake-myhra-trees.mjs next.');
