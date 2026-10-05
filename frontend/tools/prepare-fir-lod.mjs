// Derivative of the already-downloaded Poly Haven fir_tree_01 (CC0).
import {NodeIO} from '@gltf-transform/core';
import {ALL_EXTENSIONS} from '@gltf-transform/extensions';
import {weld,simplify,prune,textureCompress} from '@gltf-transform/functions';
import {MeshoptSimplifier} from 'meshoptimizer';
import sharp from 'sharp';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
const raw=process.argv[2];if(!raw)throw Error('Pass the existing MyhraWebAssets/fir directory');
const io=new NodeIO().registerExtensions(ALL_EXTENSIONS),doc=await io.read(path.join(raw,'selected.gltf'));
const twig=doc.getRoot().listMaterials().find(m=>m.getName().includes('twig'));
const diffuse=await sharp(twig.getBaseColorTexture().getImage()).removeAlpha().raw().toBuffer({resolveWithObject:true});
const alpha=await sharp(await readFile(path.join(raw,'twig-alpha.png'))).resize(diffuse.info.width,diffuse.info.height).extractChannel(0).raw().toBuffer();
twig.getBaseColorTexture().setImage(await sharp(diffuse.data,{raw:diffuse.info}).joinChannel(alpha,{raw:{width:diffuse.info.width,height:diffuse.info.height,channels:1}}).webp({quality:88}).toBuffer()).setMimeType('image/webp');
twig.setAlphaMode('MASK').setAlphaCutoff(.18).setDoubleSided(true);
await MeshoptSimplifier.ready;
await doc.transform(prune(),weld(),simplify({simplifier:MeshoptSimplifier,ratio:.008,error:.025}),prune(),textureCompress({encoder:sharp,targetFormat:'webp',resize:[1024,1024],quality:85}));
await io.write('public/courses/grenland/myhra/fir-near.glb',doc);
console.log(JSON.stringify({triangles:doc.getRoot().listMeshes().reduce((s,m)=>s+m.listPrimitives().reduce((s,p)=>s+p.getIndices().getCount()/3,0),0)}));
