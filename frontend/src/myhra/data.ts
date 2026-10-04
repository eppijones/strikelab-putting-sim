import {heightAt,type Region,type TileSpec,type Vec3,type World} from '../course/engine';
import {loadTile} from '../course/terrain';

export const ASSETS='/courses/grenland/myhra/';
export interface HoleData {
 name:string;par:number;distance:number;tee:Vec3;pin:Vec3;
 terrain:TileSpec;detail:TileSpec;regions:Region[];trees:[number,number,number,number,number][];
 pond:{points:[number,number][];height:number};road:Vec3[];
}
export interface Hole {data:HoleData;world:World}
export async function loadHole(signal?:AbortSignal):Promise<Hole>{
 const response=await fetch(ASSETS+'hole.json',{signal});
 if(!response.ok)throw new Error('The hole could not be downloaded.');
 const data:HoleData=await response.json();
 const [terrain,detail]=await Promise.all([loadTile(data.terrain,signal),loadTile(data.detail,signal)]);
 const world:World={terrain,detail,pin:data.pin,hazards:data.regions.filter(r=>r.kind==='water'),stimp:10,wind:[0,0]};
 data.tee[1]=heightAt(world,data.tee[0],data.tee[2]);
 data.pin[1]=heightAt(world,data.pin[0],data.pin[2]);
 return {data,world};
}
export {SAVE,freshHole,restoreHole,type SavedHole} from './round';
