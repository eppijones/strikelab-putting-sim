// Visual-only derivatives: no playing surface or shot physics changes.
import {NodeIO} from '@gltf-transform/core';
import {ALL_EXTENSIONS} from '@gltf-transform/extensions';
import {textureCompress} from '@gltf-transform/functions';
import sharp from 'sharp';
const io=new NodeIO().registerExtensions(ALL_EXTENSIONS),root='public/courses/grenland/myhra/';
for(const [name,size] of [['golfer-motions',1024],['cart',1024],['birch',1024],['fir-near',512]]){
 const doc=await io.read(root+name+'.glb');
 await doc.transform(textureCompress({encoder:sharp,targetFormat:'webp',resize:[size,size],quality:90}));
 await io.write(root+name+'-mobile.glb',doc);
 console.log(name,size,doc.getRoot().listTextures().length+' textures');
}
