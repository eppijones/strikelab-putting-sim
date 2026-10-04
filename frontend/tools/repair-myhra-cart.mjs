import {NodeIO} from '@gltf-transform/core';
import {ALL_EXTENSIONS} from '@gltf-transform/extensions';
const io=new NodeIO().registerExtensions(ALL_EXTENSIONS),path='public/courses/grenland/myhra/cart.glb',doc=await io.read(path);
if(doc.getRoot().listMaterials().some(m=>m.getName()==='Clear windscreen'))throw Error('Windscreen already repaired. Re-optimize the source before running again.');
let count=0;
for(const mesh of doc.getRoot().listMeshes())for(const p of mesh.listPrimitives()){
 const positions=p.getAttribute('POSITION'),indices=p.getIndices(),glass=[],body=[];
 for(let i=0;i<indices.getCount();i+=3){const ids=[indices.getScalar(i),indices.getScalar(i+1),indices.getScalar(i+2)],center=[0,0,0];for(const id of ids){const pos=positions.getElement(id,[]);pos.forEach((v,k)=>center[k]+=v/3);}
  const [x,y,z]=center,isGlass=Math.abs(x)<.164&&y>.359&&y<.611&&Math.abs(z-(.43-.249*y))<.024;
  (isGlass?glass:body).push(...ids);
 }
 if(!glass.length)continue;
 const glassPrimitive=doc.createPrimitive().setMode(p.getMode());for(const semantic of p.listSemantics())glassPrimitive.setAttribute(semantic,p.getAttribute(semantic));
 glassPrimitive.setIndices(doc.createAccessor().setType('SCALAR').setArray(new Uint32Array(glass)).setBuffer(indices.getBuffer()));
 glassPrimitive.setMaterial(doc.createMaterial('Clear windscreen').setBaseColorFactor([.83,.95,.98,.09]).setMetallicFactor(0).setRoughnessFactor(.14).setDoubleSided(true).setAlphaMode('BLEND'));
 indices.setArray(new Uint32Array(body));mesh.addPrimitive(glassPrimitive);count+=glass.length/3;
}
if(count<100)throw Error('Windscreen segmentation unexpectedly small: '+count);
await io.write(path,doc);console.log(JSON.stringify({glassTriangles:count,path}));
