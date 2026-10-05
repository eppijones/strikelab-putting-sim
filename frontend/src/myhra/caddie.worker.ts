import {advise} from './caddie';
import type {Vec3,World} from '../course/engine';
import type {PuttRange} from './shot';
self.onmessage=(e:MessageEvent<{id:number|string;world:World;ball:Vec3;puttRange?:PuttRange}>)=>{
 try{self.postMessage({id:e.data.id,advice:advise(e.data.world,e.data.ball,{puttRange:e.data.puttRange})});}catch(error){self.postMessage({id:e.data.id,error:String(error)});}
};
