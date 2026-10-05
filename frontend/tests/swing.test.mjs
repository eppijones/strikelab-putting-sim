import test from 'node:test';
import assert from 'node:assert/strict';
import {SwingController} from '../src/course/swing.ts';

test('mouse back and intentional forward crossing delivers one stroke', () => {
  const s = new SwingController(); s.begin(0, 'mouse'); s.move(1, 0, 800);
  s.move(.5, 0, 970);
  const strike = s.move(-.035, 0, 1160);
  assert.equal(strike.label, 'PURE'); assert.equal(strike.power, 1); assert.equal(strike.contact, 1);
  assert.equal(s.move(-.5, 0, 1200), undefined); assert.equal(s.release(1250), undefined);
});

test('touch release uses displayed pull, not peak or time held', () => {
  for (const held of [100, 700, 2500]) {
    const s = new SwingController(); s.begin(0, 'touch'); s.move(.9, 0, 80); s.move(.42, 0, held);
    const strike = s.release(held + 20);
    assert.equal(strike.power, .42); assert.equal(strike.tempo, 0); assert.equal(strike.label, 'PURE');
    assert.equal(s.release(held + 30), undefined);
  }
});

test('touch tap and explicit cancellation never commit a stroke', () => {
  const s = new SwingController(); s.begin(0, 'touch'); s.move(.01, 0, 100);
  assert.equal(s.release(120), undefined); assert.equal(s.phase, 'ready');
  s.begin(200, 'touch'); s.move(.8, 0, 250); s.cancel();
  assert.equal(s.release(400), undefined); assert.equal(s.phase, 'ready');
});

test('mouse release or neutral return does not masquerade as through-swing', () => {
  const s = new SwingController(); s.begin(0); s.move(.8, 0, 600); s.move(0, 0, 900);
  assert.equal(s.phase, 'downswing'); assert.equal(s.release(910), undefined);
  assert.equal(s.phase, 'ready');
});

test('path and tempo affect face and contact without random scatter', () => {
  const s = new SwingController(); s.begin(0); s.move(.8, 0, 600); s.move(.4, .5, 650);
  const hit = s.move(-.035, .5, 700);
  assert.equal(hit.label, 'FAST'); assert.ok(hit.contact < 1); assert.notEqual(hit.face, 0);
  assert.equal(hit.power, .8);
});

test('invalid and stale samples cannot poison a gesture or create a shot', () => {
  const s = new SwingController(); s.begin(0); s.move(.6, 0, 100);
  assert.equal(s.move(-.5, 0, 90), undefined); assert.equal(s.amount, .6);
  assert.equal(s.move(NaN, 0, 200), undefined); assert.equal(s.phase, 'ready');
  assert.equal(s.press(Infinity), undefined); assert.equal(s.phase, 'ready');
});

test('exactly three presses start, set power and set accuracy', () => {
  const s = new SwingController(); s.reset('three-click');
  assert.equal(s.press(0), undefined); assert.equal(s.phase, 'power');
  assert.equal(s.press(1350), undefined); assert.equal(s.phase, 'path');
  const hit = s.press(1350 + 312.5);
  assert.equal(hit.label, 'PURE'); assert.equal(hit.power, 1);
  assert.ok(Math.abs(hit.face) < 1e-10); assert.equal(s.phase, 'finish');
  assert.equal(s.press(2000), undefined);
});

test('controller must neutral-rearm on connection and after an interruption', () => {
  const s = new SwingController();
  s.controllerMove(1, 0, 0); assert.equal(s.phase, 'ready');
  s.controllerMove(0, 0, 10); s.controllerMove(0, 0, 150);
  s.controllerMove(.5, 0, 200); assert.equal(s.phase, 'backswing');
  s.controllerMove(1, 0, 500); s.controllerMove(0, 0, 600);
  assert.equal(s.phase, 'downswing'); // Spring release alone is not a hit.
  s.controllerMove(0, 0, 620, false); assert.equal(s.phase, 'ready');
  s.controllerMove(1, 0, 700); assert.equal(s.phase, 'ready');
  s.controllerMove(0, 0, 800); s.controllerMove(0, 0, 950);
  s.controllerMove(.6, 0, 1000); s.controllerMove(1, 0, 1300);
  const hit = s.controllerMove(-.14, 0, 1660);
  assert.equal(hit.power, 1); assert.equal(hit.source, 'controller');
});

test('controller polling never steals an active touch gesture', () => {
  const s = new SwingController(); s.begin(0, 'touch'); s.move(.5, 0, 100);
  s.controllerMove(.9, .5, 150); s.controllerMove(0, 0, 200, false);
  assert.equal(s.source, 'touch'); assert.equal(s.phase, 'backswing');
  assert.equal(s.release(250).power, .5);
});

test('timestamp-interpolated crossing is independent of sample frequency', () => {
  const run = hz => {
    const s = new SwingController(); s.begin(0); s.move(1, 0, 800);
    let hit;
    for (let t = 800 + 1000 / hz; t <= 1250; t += 1000 / hz) {
      hit = s.move(1 - (t - 800) / 360 * 1.035, .15, t);
      if (hit) return hit;
    }
    throw new Error('No strike');
  };
  const runs = [30, 60, 120].map(run);
  for (const hit of runs) {
    assert.equal(hit.power, 1);
    assert.ok(Math.abs(hit.tempo) < 1e-12);
    assert.ok(Math.abs(hit.face - .6) < 1e-12);
  }
});


test('cancel cannot undo an accepted stroke; settle reset explicitly rearms', () => {
  const s = new SwingController(); s.begin(0, 'touch'); s.move(.75, 0, 300); s.release(400);
  s.cancel(); assert.equal(s.phase, 'finish'); assert.equal(s.peak, .75);
  s.controllerMove(0, 0, 500, false); assert.equal(s.phase, 'finish');
  s.reset(); assert.equal(s.phase, 'ready'); assert.equal(s.peak, 0);
});
