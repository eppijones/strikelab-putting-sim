import test from 'node:test';
import assert from 'node:assert/strict';
import {Vector3} from 'three';
import {addressClub,clubFrame,golfPose,BALL_RADIUS} from '../src/myhra/golfMotion.ts';
import {blocksTeeCorridor} from '../src/course/teeClearance.ts';

test('all clubs address the back of the ball with a square striking face and a fixed length',()=>{
 for(let club=0;club<14;club++)for(const impact of [false,true]){const p=addressClub(club,impact),q=clubFrame(p.shaft),face=new Vector3(0,0,1).applyQuaternion(q),end=new Vector3(0,-p.length,0).applyQuaternion(q).add(p.hands);
  assert.ok(face.x>.97,'Club face must point down the target line');assert.ok(end.distanceTo(p.head)<1e-6,'Grip and head must share one rigid club');
  const leading=end.x+face.x*p.faceDepth;assert.ok(leading<=-BALL_RADIUS+.001,'Club must never start on the target side of the ball');assert.ok(leading>-BALL_RADIUS-.012,'Contact gap must be small');
 }
});
test('swing poses remain finite and club orientation normalized throughout the swing',()=>{
 for(const club of [0,6,13])for(let i=0;i<=100;i++)for(const follow of [false,true]){const p=golfPose(club,follow?0:i/100,follow?i/100:0),q=clubFrame(p.shaft,p.faceTurn);assert.ok([...p.hands.toArray(),...q.toArray()].every(Number.isFinite));assert.ok(Math.abs(q.length()-1)<1e-6);}
});
test('tee corridor guard removes a misplaced opening tree without clearing the side woods',()=>{
 const hole={tee:[100,0,200],pin:[100,0,0]};assert.equal(blocksTeeCorridor(103,180,hole),true);assert.equal(blocksTeeCorridor(145,180,hole),false);assert.equal(blocksTeeCorridor(100,60,hole),false);
});
