import {BALL_RADIUS_M,heightAt,type ShotResult,type Vec3,type World} from '../course/engine.ts';

/** Display adapter only. Endpoints remain authoritative; this is never a shot
 * simulation or a source of historical carry, spin, scoring or statistics. */
export function illustrativeFlight(world:World,start:Vec3,end:Vec3,putting:boolean):ShotResult {
 const duration=3.5,path:Vec3[]=[],pathTimes:number[]=[];
 const span=Math.hypot(end[0]-start[0],end[2]-start[2]);
 for(let i=0;i<=120;i++){
  const t=i/120,x=start[0]+(end[0]-start[0])*t,z=start[2]+(end[2]-start[2])*t;
  const y=putting?heightAt(world,x,z):start[1]+(end[1]-start[1])*t+Math.sin(Math.PI*t)*Math.min(18,span*.10);
  path.push([x,y+BALL_RADIUS_M,z]);pathTimes.push(t*duration);
 }
 return {path,pathTimes,end:[end[0],end[1]+BALL_RADIUS_M,end[2]],duration,carry:0,distance:span,penalty:0,made:false,reason:'Illustrative flight between manually remembered positions',physicsVersion:'illustrative-display-only'};
}
