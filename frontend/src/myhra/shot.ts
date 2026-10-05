import {
  BALL_RADIUS_M, CLUBS, PHYSICS_VERSION, greenDeceleration, heightAt, simulate, surfaceAt,
  type Shot, type ShotResult, type Vec3, type World,
} from '../course/engine.ts';
import type {Strike, SwingSource} from '../course/swing.ts';

export const SHOT_SCHEMA_VERSION = 1;
export const INPUT_PROFILE_VERSION = 'grenland-input-2';
export const CALIBRATION_ID = 'trackman-aggregate-usga-stimp-v1';
export const PUTT_RANGES = [
  {id: 'short', label: 'Short', metres: 6},
  {id: 'medium', label: 'Medium', metres: 15},
  {id: 'long', label: 'Long', metres: 35},
] as const;
export type PuttRange = typeof PUTT_RANGES[number]['id'];
export type ShotType = 'full' | 'pitch' | 'chip' | 'putt';
export interface ShotSetup {
  club: number;
  /** Nominal strength for a preview; a supplied strike replaces it, never multiplies it. */
  power: number;
  bearing: number;
  puttRange?: PuttRange;
  shotType?: ShotType;
  friendly?: boolean;
}

/** Persist this resolved launch with the recorded outcome; replay never re-resolves input. */
export interface ResolvedLaunch extends Shot {
  schemaVersion: 1;
  physicsVersion: string;
  calibrationId: string;
  inputProfile: string;
  units: 'metres-seconds-radians-bearing-degrees-launch-rpm';
  source: SwingSource | 'preview' | 'measured' | 'estimated';
  club: number | null;
  shotType: ShotType;
  puttRange?: PuttRange;
  power: number | null;
  contact: number;
  faceErrorDeg: number;
}

export const recommendPuttRange = (metres: number): PuttRange =>
  PUTT_RANGES.find(range => range.metres >= Math.max(0, metres) * 1.3)?.id ?? 'long';
export const puttRangeMetres = (range: PuttRange = 'medium') => {
  const selected = PUTT_RANGES.find(r => r.id === range);
  if (!selected) throw new Error('Unknown putting range');
  return selected.metres;
};
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

export function resolveLaunch(world: World, ball: Vec3, setup: ShotSetup, strike?: Strike): ResolvedLaunch {
  const club = CLUBS[setup.club];
  const power = strike?.power ?? setup.power;
  if (!Number.isInteger(setup.club) || !club ||
      ![...ball, power, setup.bearing, strike?.face ?? 0, strike?.contact ?? 1, world.stimp].every(Number.isFinite) ||
      power < 0 || power > 1.08 || world.stimp < 4 || world.stimp > 18) {
    throw new Error('Invalid shot setup');
  }
  const shotType: ShotType = setup.club === 13 ? 'putt' : (setup.shotType ?? 'full');
  if ((setup.shotType !== undefined && !['full', 'pitch', 'chip', 'putt'].includes(setup.shotType)) ||
      (shotType === 'putt' && setup.club !== 13)) {
    throw new Error('Invalid shot type for this club');
  }
  const putting = shotType === 'putt';
  const range = setup.puttRange ?? 'medium';
  const faceErrorDeg = (strike?.face ?? 0) * (setup.friendly ? .4 : 1);
  // A putt's pace is the displayed strength; tempo/contact never secretly changes it.
  const contact = putting ? 1 : clamp(setup.friendly ? 1 - (1 - (strike?.contact ?? 1)) * .25 : strike?.contact ?? 1, .5, 1);
  const lie = surfaceAt(world, ball[0], ball[2]);
  const lieFactor = lie === 'rough' ? .88 : lie === 'bunker' ? .62 : 1;
  let speed: number, launch: number, spin: number;
  if (putting) {
    // Range is a visible flat-green distance, independent of where the cup happens to be.
    speed = Math.sqrt(2 * greenDeceleration(world.stimp) * puttRangeMetres(range) * power);
    launch = 0; spin = 0;
  } else {
    const speedScale = shotType === 'chip' ? .26 : shotType === 'pitch' ? .58 : 1;
    launch = shotType === 'chip' ? clamp(club.loft * .4, 9, 24) :
      shotType === 'pitch' ? clamp(club.loft * .72, 24, 43) : club.launch ?? club.loft;
    speed = club.speed * power * contact * lieFactor * speedScale;
    spin = club.spin * Math.sqrt(power) * (shotType === 'chip' ? .28 : shotType === 'pitch' ? .75 : 1) *
      (lie === 'rough' ? .55 : lie === 'bunker' ? .75 : 1);
  }
  return {
    schemaVersion: SHOT_SCHEMA_VERSION, physicsVersion: PHYSICS_VERSION, calibrationId: CALIBRATION_ID,
    inputProfile: INPUT_PROFILE_VERSION, units: 'metres-seconds-radians-bearing-degrees-launch-rpm',
    source: strike?.source ?? (strike ? 'mouse' : 'preview'), club: setup.club, shotType,
    ...(putting ? {puttRange: range} : {}), power, contact, faceErrorDeg,
    speed, bearing: setup.bearing + faceErrorDeg * Math.PI / 180, launch, spin,
  };
}

/** Launch monitors use measured SI launch data directly; no game input assistance applies. */
export function measuredLaunch(shot: Shot, source: 'measured' | 'estimated' = 'measured'): ResolvedLaunch {
  if (![shot.speed, shot.bearing, shot.launch, shot.spin].every(Number.isFinite) ||
      shot.speed <= 0 || shot.speed > 100 || shot.launch < 0 || shot.launch > 80 || Math.abs(shot.spin) > 20000) {
    throw new Error('Invalid measured launch');
  }
  return {
    ...shot, schemaVersion: SHOT_SCHEMA_VERSION, physicsVersion: PHYSICS_VERSION, calibrationId: CALIBRATION_ID,
    inputProfile: 'measured-launch-v1', units: 'metres-seconds-radians-bearing-degrees-launch-rpm',
    source, club: null, shotType: shot.launch === 0 ? 'putt' : 'full', power: null, contact: 1, faceErrorDeg: 0,
  };
}

export function simulateResolved(world: World, ball: Vec3, launch: ResolvedLaunch): ShotResult {
  if (launch.schemaVersion !== SHOT_SCHEMA_VERSION || launch.physicsVersion !== PHYSICS_VERSION) {
    throw new Error('Use the recorded outcome to replay an older physics version');
  }
  if (launch.units !== 'metres-seconds-radians-bearing-degrees-launch-rpm' ||
      ![...ball, launch.speed, launch.bearing, launch.launch, launch.spin, world.stimp, ...world.wind].every(Number.isFinite) ||
      launch.speed < 0 || launch.speed > 100 || launch.launch < 0 || launch.launch > 80 ||
      Math.abs(launch.spin) > 20000 || world.stimp < 4 || world.stimp > 18) {
    throw new Error('Invalid resolved launch');
  }
  const centre: Vec3 = [ball[0], heightAt(world, ball[0], ball[2]) + BALL_RADIUS_M, ball[2]];
  if (launch.speed === 0) return {
    end: centre, path: [centre], pathTimes: [0], made: false, penalty: 0, reason: 'No stroke',
    carry: 0, distance: 0, duration: 0, apex: 0, physicsVersion: PHYSICS_VERSION,
  };
  return simulate(world, centre, launch);
}

export function simulateShot(world: World, ball: Vec3, setup: ShotSetup, strike?: Strike) {
  const launch = resolveLaunch(world, ball, setup, strike);
  return {launch, result: simulateResolved(world, ball, launch)};
}
