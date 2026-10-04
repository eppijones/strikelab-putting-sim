import test from 'node:test';
import assert from 'node:assert/strict';
import {SwingController} from '../src/course/swing.ts';

test('smooth straight analog swing yields full contact once',()=>{
 const s=new SwingController();s.begin(0);s.move(1,0,800);s.move(.5,0,960);
 const strike=s.move(0,0,1140);
 assert.equal(strike.label,'PURE');assert.equal(strike.power,1);assert.equal(strike.contact,1);
 assert.equal(s.move(0,0,1150),undefined);assert.equal(s.release(1160),undefined);assert.equal(s.phase,'finish');
});
test('path and tempo affect direction and contact',()=>{
 const s=new SwingController();s.begin(0);s.move(.8,0,600);s.move(.4,.5,700);const hit=s.move(0,.5,720);
 assert.equal(hit.label,'FAST');assert.ok(hit.contact<1);assert.notEqual(hit.face,0);assert.equal(hit.power,.8);
});
test('cancelled backswing and invalid input never create a shot',()=>{
 const s=new SwingController();s.begin(0);s.move(.8,0,600);assert.equal(s.release(650),undefined);assert.equal(s.phase,'ready');
 s.begin(700);assert.equal(s.move(NaN,0,800),undefined);assert.equal(s.phase,'ready');
});
test('three timed clicks after starting lock independent power, path and tempo',()=>{
 const s=new SwingController();s.reset('three-click');s.press(0);s.press(1500);s.press(1500+1000/(4*1.05));
 const hit=s.press(1500+1000/(4*1.05)+200);
 assert.equal(hit.label,'PURE');assert.equal(hit.power,1);assert.ok(Math.abs(hit.face)<1e-10);assert.equal(s.press(3000),undefined);
});
