import test from 'node:test';
import assert from 'node:assert/strict';
import {simulate, BALL_RADIUS_M, PHYSICS_VERSION} from '../src/course/engine.ts';
import {simulateShot, resolveLaunch, measuredLaunch, simulateResolved, PUTT_RANGES} from '../src/myhra/shot.ts';

function course({lie=2,gx=0,gz=0}={}) {
  const size=801,spacing=1,origin=[-400,-400];
  return {terrain:{size,spacing,origin,heights:Float32Array.from({length:size*size},(_,i)=>(i%size-400)*gx+(Math.floor(i/size)-400)*gz),lies:new Uint8Array(size*size).fill(lie)},pin:[350,0,350],stimp:10,wind:[0,0]};
}
const ball=[0,0,0], perfect={power:.5,face:0,tempo:0,contact:1,label:'PURE',source:'touch'};

test('putting strength uses a visible fixed range and one absolute input',()=>{
  const world=course();
  for(const range of PUTT_RANGES)for(const power of [.1,.5,1]) {
    const {result}=simulateShot(world,ball,{club:13,bearing:0,power,puttRange:range.id});
    assert.ok(Math.abs(result.distance-range.metres*power)<.015);
  }
  const setup={club:13,bearing:0,power:.2,puttRange:'short'};
  const actual=simulateShot(world,ball,setup,perfect).result;
  const preview=simulateShot(world,ball,{...setup,power:.5}).result;
  assert.deepEqual(actual,preview);
  assert.ok(Math.abs(actual.distance-3)<.015,'50% means half the 6m range, not .2 times .5');
});

test('moving the cup cannot silently change the same putt launch',()=>{
  const world=course(),setup={club:13,power:.5,bearing:0,puttRange:'medium'};
  const near=resolveLaunch({...world,pin:[0,0,-2]},ball,setup);
  const far=resolveLaunch({...world,pin:[0,0,-20]},ball,setup);
  assert.equal(near.speed,far.speed);
});

test('perfect delivered shot and nominal preview use identical simulation',()=>{
  const world=course({lie:1});
  for(const club of [0,6,9,11,13]) {
    const setup={club,power:.63,bearing:.12,puttRange:'medium',friendly:true};
    assert.deepEqual(simulateShot(world,ball,setup).result,simulateShot(world,ball,setup,{...perfect,power:.63}).result);
  }
});

test('forgiving controls preserve strength and putting contact never secretly changes pace',()=>{
  const world=course();
  const hit={...perfect,power:.94,contact:.8,face:2};
  const a=resolveLaunch(world,ball,{club:13,power:1,bearing:0,puttRange:'short',friendly:true},hit);
  const b=resolveLaunch(world,ball,{club:13,power:1,bearing:0,puttRange:'short',friendly:false},hit);
  assert.equal(a.power,.94);assert.equal(a.speed,b.speed);assert.equal(a.contact,1);
  assert.ok(a.faceErrorDeg<b.faceErrorDeg);
});

test('zero preview is stationary; invalid setups fail before changing a round',()=>{
  const world=course(),setup={club:13,power:0,bearing:0,puttRange:'short'};
  const {result}=simulateShot(world,ball,setup);
  assert.equal(result.distance,0);assert.equal(result.duration,0);assert.equal(result.path[0][1],BALL_RADIUS_M);
  for(const invalid of [{power:NaN},{club:-1},{power:-.1},{puttRange:'secret'},{shotType:'warp'}])assert.throws(()=>resolveLaunch(world,ball,{...setup,...invalid}));
});

test('USGA standard 6.4ft/s roll matches green-speed feet, independent reference fixture',()=>{
  // USGA Science of Golf, Putting facilitator guide (not a formula copied from the engine).
  // https://www.usga.org/content/dam/usga/pdf/science-of-golf/middle-school/Putting/putting_facilitator_guide_MS.pdf
  const world=course();
  for(const stimp of [6,8,10,12,14]) {
    const r=simulate({...world,stimp},ball,{speed:1.95072,bearing:0,launch:0,spin:0});
    assert.ok(Math.abs(r.distance-stimp*.3048)<.015);
  }
});

test('flat and planar slopes agree with analytic solid-sphere rolling and break direction',()=>{
  const speed=2,friction=1.95072**2/(2*10*.3048);
  const up=simulate(course({gz:-.02}),ball,{speed,bearing:0,launch:0,spin:0});
  const down=simulate(course({gz:.02}),ball,{speed,bearing:0,launch:0,spin:0});
  assert.ok(Math.abs(up.distance-speed**2/(2*(friction+5/7*9.81*.02)))<.025);
  assert.ok(Math.abs(down.distance-speed**2/(2*(friction-5/7*9.81*.02)))<.025);
  const left=simulate(course({gx:.02}),ball,{speed,bearing:0,launch:0,spin:0});
  const right=simulate(course({gx:-.02}),ball,{speed,bearing:0,launch:0,spin:0});
  assert.ok(left.end[0]<-.1);assert.ok(right.end[0]>.1);assert.ok(Math.abs(left.end[0]+right.end[0])<.005);
});

test('published Trackman aggregates bound carry/apex fit, not launch-monitor certification',()=>{
  // Historical published PGA averages, units MPH/degrees/RPM/metres. Aggregate
  // launch values are not one measured shot and are not Grenland calibration data.
  // https://support.trackmangolf.com/hc/en-us/article_attachments/7349802113051
  // https://www.trackman.com/blog/introducing-updated-tour-averages (links historical comparison)
  const references=[
    {club:'Driver',mph:167,launch:10.9,spin:2686,carry:251,apex:29},
    {club:'7 iron',mph:120,launch:16.3,spin:7097,carry:157,apex:29},
    {club:'PW',mph:102,launch:24.2,spin:9304,carry:124,apex:27},
  ];
  const world=course({lie:1});
  for(const fixture of references) {
    const r=simulate(world,ball,{speed:fixture.mph*.44704,bearing:0,launch:fixture.launch,spin:fixture.spin});
    assert.ok(Math.abs(r.carry-fixture.carry)/fixture.carry<.05,fixture.club+' carry');
    assert.ok(Math.abs(r.apex-fixture.apex)/fixture.apex<.10,fixture.club+' apex');
  }
});

test('recorded samples use actual increasing seconds, ball centres and identical reruns',()=>{
  const world=course({lie:1}),launch=measuredLaunch({speed:50,bearing:.3,launch:20,spin:5000});
  const a=simulateResolved(world,ball,launch),b=simulateResolved(world,ball,launch);
  assert.deepEqual(a,b);assert.equal(a.physicsVersion,PHYSICS_VERSION);
  assert.equal(a.path.length,a.pathTimes.length);assert.equal(a.pathTimes[0],0);
  assert.equal(a.pathTimes.at(-1),a.duration);assert.ok(a.duration>0);
  assert.ok(a.pathTimes.every((t,i)=>i===0||t>a.pathTimes[i-1]));
  assert.ok(a.path.every(p=>p[1]>=BALL_RADIUS_M-1e-8));
  assert.throws(()=>simulateResolved(world,ball,{...launch,physicsVersion:'old'}));
});

test('edge entries require softer pace than centre entries',()=>{
  const world=course();world.pin=[0,0,0];
  const fast={speed:1.6,bearing:0,launch:0,spin:0};
  assert.equal(simulate(world,[0,0,1],fast).made,true);
  assert.equal(simulate(world,[.027,0,1],fast).made,false);
  assert.equal(simulate(world,[.027,0,1],{...fast,speed:1.2}).made,true);
});

test('measured launches bypass game power, club and forgiving direction',()=>{
  const shot={speed:43,bearing:.45,launch:21,spin:6100},launch=measuredLaunch(shot);
  assert.equal(launch.source,'measured');assert.equal(launch.power,null);assert.equal(launch.club,null);
  for(const key of Object.keys(shot))assert.equal(launch[key],shot[key]);
  assert.throws(()=>measuredLaunch({...shot,speed:NaN}));
});
