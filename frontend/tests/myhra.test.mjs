import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {advise,speedFor} from '../src/myhra/caddie.ts';
import {freshHole,restoreHole} from '../src/myhra/round.ts';
import {CLUBS,simulate,heightAt,surfaceAt,distance} from '../src/course/engine.ts';

const root=new URL('../public/courses/grenland/',import.meta.url);
const data=JSON.parse(readFileSync(new URL('myhra/hole.json',root)));
function tile(spec){const heights=readFileSync(new URL(spec.url,root)),lies=readFileSync(new URL(spec.surfaces,root));return {...spec,heights:new Float32Array(heights.buffer.slice(heights.byteOffset,heights.byteOffset+heights.byteLength)),lies:new Uint8Array(lies)};}
const world={terrain:tile(data.terrain),detail:tile(data.detail),pin:data.pin,hazards:data.regions.filter(r=>r.kind==='water'),stimp:10,wind:[0,0]},hole={world,data};
const shot=(ball,a,power=1,angle=0)=>simulate(world,ball,{speed:speedFor(world,ball,a.club,a.power)*power,bearing:a.bearing+angle*Math.PI/180,launch:CLUBS[a.club].loft,spin:CLUBS[a.club].spin});

test('the caddie tee line leaves room for small human power and direction errors',()=>{
 const a=advise(world,data.tee);
 for(const power of [.96,1])for(const angle of [-2,0,2]){
  const r=shot(data.tee,a,power,angle);assert.equal(r.penalty,0,`power ${power}, angle ${angle}`);assert.ok(distance(r.end,data.pin)<distance(data.tee,data.pin)-100);
 }
});
test('the 4 m putting exercise has a useful caddie line and can reach the cup',()=>{
 const ball=[data.pin[0],0,data.pin[2]+4];ball[1]=heightAt(world,ball[0],ball[2]);assert.equal(surfaceAt(world,ball[0],ball[2]),'green');
 const a=advise(world,ball),r=shot(ball,a);assert.equal(a.club,13);assert.ok(r.made||distance(r.end,data.pin)<.3);assert.match(a.reason,/putt/);
});
test('the diagram-informed pond applies a penalty and returns the ball to its previous lie',()=>{
 const ball=[414,heightAt(world,414,269),269],r=simulate(world,ball,{speed:12,bearing:0,launch:0,spin:0});
 assert.equal(r.penalty,1);assert.deepEqual(r.end,ball);
});
test('saved rounds reject corrupt history and impossible scores without losing the playable tee',()=>{
 const valid={...freshHole(hole),strokes:2,history:[{club:'Driver',distance:150,carry:140,lie:'fairway',penalty:1}]};
 assert.equal(restoreHole(hole,JSON.stringify(valid)).strokes,2);
 for(const invalid of [{...valid,history:[null]},{...valid,strokes:99},{...valid,complete:true},{...valid,ball:[999,0,1]},{...valid,history:[{...valid.history[0],distance:'bad'}]}])assert.deepEqual(restoreHole(hole,JSON.stringify(invalid)),freshHole(hole));
 assert.deepEqual(restoreHole(hole,'{truncated'),freshHole(hole));
});


test('generated trees leave the Myhra tee-to-fairway opening clear, including canopy width',()=>{
 const a=data.tee,b=[355,0,405],dx=b[0]-a[0],dz=b[2]-a[2],length=Math.hypot(dx,dz);
 for(const t of data.trees){const along=((t[0]-a[0])*dx+(t[2]-a[2])*dz)/length,cross=Math.abs((t[0]-a[0])*dz-(t[2]-a[2])*dx)/length;
  if(along>=0&&along<=length)assert.ok(cross>=15+8*along/length-.02,'Tree intrudes into the verified open tee corridor');
 }
});
