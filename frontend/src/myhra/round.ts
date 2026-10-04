import {CLUBS,distance,heightAt,type Vec3} from '../course/engine.ts';
import type {Hole} from './data';

export interface SavedHole {version:1;ball:Vec3;strokes:number;complete:boolean;history:{club:string;distance:number;carry:number;lie:string;penalty:number}[]}
export const SAVE='strikelab.myhra.hole.v2';
export function freshHole(hole:Hole):SavedHole{return {version:1,ball:[...hole.data.tee],strokes:0,complete:false,history:[]};}
export function restoreHole(hole:Hole,raw:string|null):SavedHole{
 try{
  const saved=JSON.parse(raw??'null');
  if(saved?.version!==1||!Array.isArray(saved.ball)||saved.ball.length!==3||!saved.ball.every(Number.isFinite)||!Number.isInteger(saved.strokes)||saved.strokes<0||!Array.isArray(saved.history)||saved.history.length>500||typeof saved.complete!=='boolean')return freshHole(hole);
  if(saved.ball[0]<0||saved.ball[0]>768||saved.ball[2]<0||saved.ball[2]>768)return freshHole(hole);
  let strokes=0;
  for(const h of saved.history){
   if(!h||!CLUBS.some(c=>c.name===h.club)||!['rough','fairway','green','bunker'].includes(h.lie)||![0,1].includes(h.penalty)||![h.distance,h.carry].every(n=>Number.isFinite(n)&&n>=0&&n<2000))return freshHole(hole);
   strokes+=1+h.penalty;
  }
  if(saved.strokes!==strokes||(saved.complete&&(strokes===0||distance(saved.ball,hole.data.pin)>.15)))return freshHole(hole);
  // The terrain is authoritative if a cached round spans a course-art update.
  return {...saved,ball:saved.complete?[...hole.data.pin]:[saved.ball[0],heightAt(hole.world,saved.ball[0],saved.ball[2]),saved.ball[2]]};
 }catch{return freshHole(hole);}
}
