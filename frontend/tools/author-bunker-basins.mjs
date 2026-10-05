import {readFile,writeFile} from 'node:fs/promises';
const root='public/courses/grenland/',file=root+'myhra/hole.json',data=JSON.parse(await readFile(file));
const regions=data.regions.filter(r=>r.kind==='bunker'&&r.points.some(([x,z])=>x>330&&x<440&&z>215&&z<435));
const inside=(points,x,z)=>{let yes=false;for(let i=0,j=points.length-1;i<points.length;j=i++){const a=points[i],b=points[j];if((a[1]>z)!==(b[1]>z)&&x<(b[0]-a[0])*(z-a[1])/(b[1]-a[1])+a[0])yes=!yes;}return yes;};
const edgeDistance=(points,x,z)=>Math.min(...points.map((a,i)=>{const b=points[(i+1)%points.length],dx=b[0]-a[0],dz=b[1]-a[1],t=Math.max(0,Math.min(1,((x-a[0])*dx+(z-a[1])*dz)/(dx*dx+dz*dz)));return Math.hypot(x-a[0]-t*dx,z-a[1]-t*dz);}));
const smooth=t=>{t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);};
for(const [key,source,target] of [['terrain','myhra/terrain.f32','myhra/terrain-basins.f32'],['detail','myhra/green.f32','myhra/green-basins.f32']]){
 const spec=data[key],buffer=await readFile(root+source),heights=new Float32Array(buffer.buffer.slice(buffer.byteOffset,buffer.byteOffset+buffer.byteLength));let changed=0;
 for(let iz=0;iz<spec.size;iz++)for(let ix=0;ix<spec.size;ix++){const x=spec.origin[0]+ix*spec.spacing,z=spec.origin[1]+iz*spec.spacing;if(x<328||x>442||z<213||z>437)continue;
  for(const r of regions){const distance=edgeDistance(r.points,x,z),contained=inside(r.points,x,z);if(distance>1.2&&!contained)continue;const depth=contained?-.32*smooth(distance/1.6):.055*(1-smooth(distance/1.2));heights[iz*spec.size+ix]+=depth;changed++;}
 }
 await writeFile(root+target,new Uint8Array(heights.buffer));data[key].url=target;console.log(key,changed,'authored basin/lip vertices');
}
data.surfaceAuthorship={bunkers:'Authored 0.32 m basins and 0.055 m lips on existing candidate footprints, not measured bunker surveys',green:'Existing approximate detail surface; no surveyed green-contour claim'};
await writeFile(file,JSON.stringify(data));
