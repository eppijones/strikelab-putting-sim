export type SwingMode = 'analog' | 'three-click';
export type SwingSource = 'mouse' | 'touch' | 'controller' | 'timing';
export type SwingPhase = 'ready' | 'backswing' | 'downswing' | 'power' | 'path' | 'tempo' | 'finish';
export interface Strike {
  power: number; face: number; tempo: number; contact: number; label: string;
  source?: SwingSource;
}
export interface SwingView {
  phase: SwingPhase; amount: number; needle: number; path: number; feedback: string;
  peak: number; lockedPower: number;
  source?: SwingSource;
}
const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));

/** Pure input state. Positions are normalized; timestamps are monotonic milliseconds. */
export class SwingController {
  mode: SwingMode = 'analog';
  phase: SwingPhase = 'ready';
  source: SwingSource = 'mouse';
  amount = 0; needle = 0; path = 0;
  started = 0; peakAt = 0; peak = 0; lockedPower = 0; lockedFace = 0;
  feedback = 'Pull back, then swing through';
  private lastAt = -Infinity;
  private lastPull = 0;
  private lastSide = 0;
  private controllerArmed = false;
  private controllerNeutralAt: number | null = null;

  reset(mode = this.mode) {
    this.mode = mode; this.phase = 'ready'; this.source = 'mouse';
    this.amount = 0; this.needle = 0; this.path = 0; this.peak = 0;
    this.lockedPower = 0; this.lockedFace = 0; this.lastAt = -Infinity;
    this.lastPull = 0; this.lastSide = 0;
    this.controllerArmed = false; this.controllerNeutralAt = null;
    this.feedback = mode === 'analog' ? 'Pull back, then swing through' : 'Three clicks · start, power, accuracy';
  }

  cancel() {
    // A committed stroke belongs to the round/animation until an explicit settle reset.
    if (this.phase !== 'finish') this.reset();
  }

  begin(now: number, source: SwingSource = 'mouse') {
    if (this.phase !== 'ready' || !Number.isFinite(now)) return;
    this.source = source; this.started = now; this.lastAt = now;
    this.peakAt = now; this.peak = 0; this.lastPull = 0; this.lastSide = 0;
    this.phase = 'backswing';
    this.feedback = source === 'touch' ? 'Pull to set power · release to hit' : 'Smooth backswing';
  }

  move(pull: number, side: number, now: number): Strike | undefined {
    if (![pull, side, now].every(Number.isFinite)) { this.cancel(); return; }
    if (this.phase !== 'backswing' && this.phase !== 'downswing') return;
    if (now < this.lastAt) return;
    const previousAt = this.lastAt, previousPull = this.lastPull, previousSide = this.lastSide;
    this.lastAt = now; this.lastPull = pull; this.lastSide = side;
    this.path = clamp(side, -1, 1); this.amount = clamp(pull, 0, 1);
    if (this.amount > this.peak) { this.peak = this.amount; this.peakAt = now; }
    if (this.source === 'touch') {
      // A touch player can shorten the selected pull before releasing. No hidden tempo test.
      this.feedback = this.amount < .025 ? 'Pull down to set power' : 'Release to hit · move aside to cancel';
      return;
    }
    if (this.peak > .08 && this.amount < this.peak - .06) {
      this.phase = 'downswing'; this.feedback = 'Swing forward through the ball';
    }
    // A returning controller spring stops at neutral. Require an intentional forward crossing.
    const crossing = this.source === 'controller' ? -.14 : -.035;
    if (this.phase === 'downswing' && pull <= crossing && previousPull > crossing) {
      const fraction = clamp((crossing - previousPull) / (pull - previousPull), 0, 1);
      this.path = clamp(previousSide + (side - previousSide) * fraction, -1, 1);
      return this.finish(previousAt + (now - previousAt) * fraction);
    }
  }

  release(now: number): Strike | undefined {
    if (this.phase === 'finish' || this.phase === 'ready') return;
    if (!Number.isFinite(now) || now < this.lastAt) { this.cancel(); return; }
    if (this.source === 'touch' && this.phase === 'backswing' && this.amount >= .025) {
      return this.strike(this.amount, this.path * 2, 0);
    }
    // Mouse/controller release is cancellation; only an intentional forward crossing hits.
    this.cancel();
  }

  finish(now: number): Strike {
    const ideal = 220 + this.peak * 140;
    const tempo = clamp(((now - this.peakAt) - ideal) / ideal, -1, 1);
    return this.strike(this.peak, this.path * 4 + tempo * 1.5, tempo);
  }

  /** Feed every frame, including neutral and disconnect. No caller-side axis clipping. */
  controllerMove(pull: number, side: number, now: number, connected = true): Strike | undefined {
    if (!connected || ![pull, side, now].every(Number.isFinite)) {
      this.controllerArmed = false; this.controllerNeutralAt = null;
      if (this.source === 'controller' && this.phase !== 'finish') this.cancel();
      return;
    }
    if (this.phase !== 'ready') {
      if (this.source === 'controller') return this.move(pull, side, now);
      return;
    }
    if (Math.abs(pull) < .12 && Math.abs(side) < .15) {
      this.controllerNeutralAt ??= now;
      if (now - this.controllerNeutralAt >= 120) this.controllerArmed = true;
      return;
    }
    this.controllerNeutralAt = null;
    if (!this.controllerArmed || pull < .16) return;
    this.controllerArmed = false;
    this.begin(now, 'controller');
    return this.move(pull, side, now);
  }

  press(now: number): Strike | undefined {
    if (!Number.isFinite(now) || now < this.lastAt) return;
    this.lastAt = now;
    if (this.phase === 'ready') {
      this.source = 'timing'; this.phase = 'power'; this.started = now;
      this.feedback = 'Click 2 · set power'; return;
    }
    this.update(now);
    if (this.phase === 'power') {
      this.lockedPower = this.amount; this.phase = 'path'; this.started = now;
      this.needle = -1; this.feedback = 'Click 3 · stop in the centre'; return;
    }
    if (this.phase === 'path') return this.strike(this.lockedPower, this.needle * 4, this.needle * .5);
  }

  update(now: number) {
    if (!Number.isFinite(now)) return;
    const elapsed = Math.max(0, now - this.started) / 1000;
    if (this.phase === 'power') {
      const cycle = (elapsed / 1.35) % 2;
      this.amount = cycle <= 1 ? cycle : 2 - cycle;
    }
    if (this.phase === 'path') this.needle = Math.sin(elapsed * Math.PI * 2 * .8 - Math.PI / 2);
  }

  strike(power: number, face: number, tempo: number): Strike {
    const contact = clamp(1 - Math.abs(face) * .012 - Math.abs(tempo) * .035, .82, 1);
    const label = Math.abs(face) < .8 && Math.abs(tempo) < .25 ? 'PURE' :
      Math.abs(tempo) > .6 ? (tempo < 0 ? 'FAST' : 'SLOW') : face < -1 ? 'PULLED' : face > 1 ? 'PUSHED' : 'SOLID';
    this.phase = 'finish'; this.amount = clamp(power, .01, 1); this.feedback = label;
    return { power: this.amount, face, tempo, contact, label, source: this.source };
  }

  view(): SwingView {
    return { phase: this.phase, amount: this.amount, needle: this.needle, path: this.path, feedback: this.feedback, source: this.source, peak:this.peak, lockedPower:this.lockedPower };
  }
}
