import type {SwingMode} from '../course/swing';
export type Quality='economy'|'balanced'|'high';
export interface Settings {version:1;quality:Quality;swingMode:SwingMode;sound:boolean;friendly:boolean;prediction:boolean;grid:boolean;hour:number;dayCycle:boolean}
export const SETTINGS_KEY='strikelab.myhra.settings.v1';
export function readSettings():Settings{
 const defaults:Settings={version:1,quality:matchMedia('(pointer:coarse)').matches?'balanced':'high',swingMode:'analog',sound:true,friendly:true,prediction:true,grid:true,hour:10,dayCycle:false};
 try{
  const v=JSON.parse(localStorage.getItem(SETTINGS_KEY)??'null');if(!v||v.version!==1)return defaults;
  return {...defaults,quality:['economy','balanced','high'].includes(v.quality)?v.quality:defaults.quality,swingMode:v.swingMode==='three-click'?'three-click':'analog',
   ...Object.fromEntries(['sound','friendly','prediction','grid','dayCycle'].filter(k=>typeof v[k]==='boolean').map(k=>[k,v[k]])),hour:Number.isFinite(v.hour)&&v.hour>=6.5&&v.hour<=19.5?v.hour:10};
 }catch{return defaults;}
}
