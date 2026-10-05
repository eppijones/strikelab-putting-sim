import test from 'node:test';import assert from 'node:assert/strict';
import {createShotContext,createShotIntent} from '../src/myhra/contracts.ts';
import {measuredLaunch} from '../src/myhra/shot.ts';
import {advanceCart,samplePath} from '../src/myhra/sceneMotion.ts';
import {illustrativeFlight} from '../src/myhra/replay.ts';
const size=101,world={terrain:{size,spacing:1,origin:[0,0],heights:new Float32Array(size*size),lies:new Uint8Array(size*size).fill(1)},pin:[50,0,20],stimp:10,wind:[0,0]};
test('context names ground contact separately from a ball centre and measured assistance stays off',()=>{
 const context=createShotContext(world,[50,.02135,50],'example-revision');assert.deepEqual(context.ballGround,[50,0,50]);
 const launch=measuredLaunch({speed:45,bearing:0,launch:18,spin:5000});const intent=createShotIntent(launch,true);assert.equal(intent.assistance.enabled,false);assert.deepEqual(intent.launch,launch);
});
test('cart travel and braking are consistent at 30/60/120 Hz; boost is controlled',()=>{
 const runs=[30,60,120].map(hz=>{const car={x:50,z:80,heading:0,speed:0,steer:0};for(let i=0;i<hz*2;i++)advanceCart(car,{forward:1,turn:0,boost:true},1/hz,world,new Map(),true);const top=car.speed;for(let i=0;i<hz;i++)advanceCart(car,undefined,1/hz,world,new Map(),false);return {car,top};});
 for(const r of runs){assert.ok(r.top<=9.5);assert.ok(Math.abs(r.car.speed)<.04);assert.ok(Math.abs(r.car.z-runs[0].car.z)<.001);}
});
test('camera sampling uses sample times rather than treating every sample as equally timed',()=>{
 assert.deepEqual(samplePath([[0,0,0],[1,0,0],[4,0,0]],1,4,[0,1,4]),[1,0,0]);
 assert.deepEqual(samplePath([[0,0,0],[1,0,0],[4,0,0]],2,4,[0,1,4]),[2,0,0]);
});
test('illustrative 3D replay retains remembered endpoints without inventing carry or cup outcomes',()=>{
 const start=[50,0,60],end=[50,0,20],result=illustrativeFlight(world,start,end,false);
 assert.equal(result.physicsVersion,'illustrative-display-only');assert.equal(result.carry,0);assert.equal(result.made,false);assert.deepEqual(result.path[0],[50,.02135,60]);assert.deepEqual(result.end,[50,.02135,20]);assert.deepEqual(start,[50,0,60]);
});
