import {heightAt,surfaceAt,type Vec3,type World} from '../course/engine.ts';
import type {TravelInput} from '../course/useDriveInput';

export interface CartState {x:number;z:number;heading:number;speed:number;steer:number}
export type TreeIndex=Map<string,[number,number][]>;
export function treeCollisionIndex(trees:number[][]):TreeIndex {
 const index:TreeIndex=new Map();
 for(const t of trees){const key=`${Math.floor(t[0]/12)},${Math.floor(t[2]/12)}`;const bucket=index.get(key)??[];bucket.push([t[0],t[2]]);index.set(key,bucket);}
 return index;
}
export function cartClear(world:World,trees:TreeIndex,x:number,z:number,heading:number,previousY:number){
 const fx=Math.sin(heading),fz=-Math.cos(heading),rx=Math.cos(heading),rz=Math.sin(heading);
 // Check the body footprint, not only its centre, before crossing a bank or trunk.
 for(const forward of [-.95,0,.95])for(const lateral of [-.66,.66]){
  const px=x+fx*forward+rx*lateral,pz=z+fz*forward+rz*lateral,lie=surfaceAt(world,px,pz);
  if(['water','out','green','bunker'].includes(lie)||Math.abs(heightAt(world,px,pz)-previousY)>.7)return false;
  for(let iz=-1;iz<=1;iz++)for(let ix=-1;ix<=1;ix++)for(const t of trees.get(`${Math.floor(px/12)+ix},${Math.floor(pz/12)+iz}`)??[])if(Math.hypot(t[0]-px,t[1]-pz)<.47)return false;
 }
 return true;
}
/** Fixed-size steps make braking and reverse behave the same at 30 and 60 Hz. */
export function advanceCart(car:CartState,input:TravelInput|undefined,seconds:number,world:World,trees:TreeIndex,enabled:boolean){
 let remaining=Math.min(.1,Math.max(0,seconds));
 while(remaining>1e-8){const dt=Math.min(1/120,remaining);remaining-=dt;
  const throttle=enabled?(input?.forward??0):0,target=throttle*(throttle<0?3.5:input?.boost?9.5:6.5);
  const braking=Math.abs(throttle)<.03||car.speed*throttle<0,rate=braking?10:2.2;
  car.speed+=(target-car.speed)*(1-Math.exp(-rate*dt));if(Math.abs(car.speed)<.035)car.speed=0;
  car.steer+=((enabled?(input?.turn??0):0)-car.steer)*(1-Math.exp(-dt*8));
  const previousHeading=car.heading;
  // A bicycle model limits steering at speed and reverses it when backing up.
  car.heading+=Math.tan(car.steer*.48/(1+Math.abs(car.speed)*.07))*car.speed/1.7*dt;
  const x=car.x+Math.sin(car.heading)*car.speed*dt,z=car.z-Math.cos(car.heading)*car.speed*dt;
  if(cartClear(world,trees,x,z,car.heading,heightAt(world,car.x,car.z))){car.x=x;car.z=z;}else{car.speed=0;car.heading=previousHeading;}
 }
}
export function slopeAt(world:World,x:number,z:number){
 const d=.25;return [(heightAt(world,x+d,z)-heightAt(world,x-d,z))/(2*d),(heightAt(world,x,z+d)-heightAt(world,x,z-d))/(2*d)] as [number,number];
}
export function samplePath(path:Vec3[],time:number,duration:number,pathTimes?:number[]):Vec3 {
 if(pathTimes?.length===path.length){let lo=0,hi=pathTimes.length-1;while(lo+1<hi){const mid=(lo+hi)>>1;if(pathTimes[mid]<=time)lo=mid;else hi=mid;}const f=Math.max(0,Math.min(1,(time-pathTimes[lo])/(pathTimes[hi]-pathTimes[lo]||1))),a=path[lo],b=path[hi];return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];}
 const step=Math.min(path.length-1,Math.max(0,time)/Math.max(.001,duration)*(path.length-1));
 const i=Math.floor(step),a=path[i],b=path[Math.min(i+1,path.length-1)],f=step-i;
 return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];
}
