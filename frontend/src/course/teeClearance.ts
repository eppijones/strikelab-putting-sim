import type {PracticeHole} from './engine.ts';

/** The generated tree layer must leave an open first-shot corridor from each tee.
 * Side woods remain; this is a conservative guard, not a claim of surveyed trees.
 */
export function blocksTeeCorridor(x:number,z:number,hole:Pick<PracticeHole,'tee'|'pin'|'route'>){
 const [tx,,tz]=hole.tee,target=hole.route?.[1]??[hole.pin[0],hole.pin[2]],dx=target[0]-tx,dz=target[1]-tz,length=Math.hypot(dx,dz),reach=Math.min(105,length*.7);
 if(length<1)return false;
 const along=((x-tx)*dx+(z-tz)*dz)/length,across=Math.abs((x-tx)*dz-(z-tz)*dx)/length;
 return along>-12&&along<reach&&across<12+Math.max(0,along)*.035;
}
