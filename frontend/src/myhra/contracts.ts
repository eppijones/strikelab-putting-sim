import {PHYSICS_VERSION,heightAt,type ShotResult,type Vec3,type World} from '../course/engine.ts';
import type {ResolvedLaunch} from './shot';
export const CONTRACT_VERSION=1;

/** +X east across the source tile, +Y up, +Z south; metres and seconds.
 * A position named ground is contact with terrain. Outcome path/end are sphere
 * centres, including one ball radius. Bearings: 0 points -Z, positive towards +X. */
export interface ShotContext {
 schemaVersion:1;courseRevision:string;playingSurface:'myhra-authoritative-grid';physicsVersion:string;
 ballGround:Vec3;pinGround:Vec3;conditions:{stimpFeet:number;windMetresPerSecond:[number,number]};
}
export interface ShotIntent {
 schemaVersion:1;source:ResolvedLaunch['source'];club:number|null;shotType:ResolvedLaunch['shotType'];
 launch:ResolvedLaunch;assistance:{enabled:boolean;faceScale:number;contactProtection:number};
}
export interface ShotOutcome {schemaVersion:1;physicsVersion:string;result:ShotResult}
export type {JournalRound as RoundRecord,WhatIfRecord as WhatIfScenario} from './journal';
export function createShotContext(world:World,ball:Vec3,courseRevision:string):ShotContext {
 return {schemaVersion:1,courseRevision,playingSurface:'myhra-authoritative-grid',physicsVersion:PHYSICS_VERSION,ballGround:[ball[0],heightAt(world,ball[0],ball[2]),ball[2]],pinGround:[world.pin[0],heightAt(world,world.pin[0],world.pin[2]),world.pin[2]],conditions:{stimpFeet:world.stimp,windMetresPerSecond:[...world.wind]}};
}
export function createShotIntent(launch:ResolvedLaunch,friendly:boolean):ShotIntent {
 const enabled=friendly&&launch.source!=='measured'&&launch.source!=='estimated';
 return {schemaVersion:1,source:launch.source,club:launch.club,shotType:launch.shotType,launch,assistance:{enabled,faceScale:enabled?.4:1,contactProtection:enabled?.75:0}};
}
export function createShotOutcome(result:ShotResult):ShotOutcome {return {schemaVersion:1,physicsVersion:result.physicsVersion??PHYSICS_VERSION,result};}
